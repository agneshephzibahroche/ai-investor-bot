# src/webapp/routes.py

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import render_template, request, flash, redirect, url_for, current_app
from newsapi import NewsApiClient

from src.webapp.app import app
from src.webapp.models import db, Query
from src.inference import ROOT_DIR, load_resources, fetch_and_prepare, recursive_forecast
from src.advisor   import get_signal, format_advice, compute_last_indicators

# ---------- USE LOCAL MODELS TO POPULATE TICKERS ----------
MODEL_DIR = os.path.join(ROOT_DIR, "models")

def get_available_tickers():
    """Return tickers that have BOTH a model and a scaler in models/."""
    if not os.path.isdir(MODEL_DIR):
        return []
    have_models = {
        fn[len("lstm_"):-len(".h5")]
        for fn in os.listdir(MODEL_DIR)
        if fn.startswith("lstm_") and fn.endswith(".h5")
    }
    have_scalers = {
        fn[len("scaler_"):-len(".save")]
        for fn in os.listdir(MODEL_DIR)
        if fn.startswith("scaler_") and fn.endswith(".save")
    }
    return sorted(have_models & have_scalers)

AVAILABLE_TICKERS = get_available_tickers()

# Init NewsAPI (optional)
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', '')
newsapi     = NewsApiClient(api_key=NEWSAPI_KEY) if NEWSAPI_KEY else None


@app.route('/', methods=['GET', 'POST'])
def index():
    today         = datetime.today().date()
    default_start = (today - timedelta(days=90)).isoformat()
    default_end   = today.isoformat()

    if request.method == 'POST':
        ticker     = request.form.get('ticker', '').upper()
        weeks      = int(request.form.get('weeks', 1))
        start_date = request.form.get('start_date')
        end_date   = request.form.get('end_date')

        if ticker not in AVAILABLE_TICKERS:
            flash(f"No model available for “{ticker}”. Please select from the dropdown.", 'danger')
            return redirect(url_for('index'))

        try:
            # --- Load scalers + model ---
            scaler_all, scaler_close, model = load_resources(ticker)

            # --- Prepare features ---
            features, full_df = fetch_and_prepare(ticker)

            # --- Forecast ---
            days  = weeks * 5
            preds = recursive_forecast(features, scaler_all, scaler_close, model, days)

            # --- Signals (force scalars) ---
            current_close = float(full_df['Close'].iloc[-1])
            forecast_end  = float(preds[-1])
            pct_change    = (forecast_end - current_close) / current_close
            signal, _     = get_signal(current_close, forecast_end, 0.02)

            # --- Indicators ---
            inds = compute_last_indicators(full_df)

            # --- News Sentiment ---
            sentiment = 0.0
            if newsapi:
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                sia      = SentimentIntensityAnalyzer()
                articles = newsapi.get_everything(
                    q=ticker,
                    from_param=(today - timedelta(days=7)).isoformat(),
                    language='en',
                    sort_by='relevancy',
                    page_size=20
                )['articles']
                scores = [sia.polarity_scores(a['title'])['compound'] for a in articles]
                sentiment = float(np.mean(scores)) if scores else 0.0

            # --- Advice with reasoning ---
            advice_text = format_advice(
                current=current_close,
                predicted=forecast_end,
                pct=pct_change,
                signal=signal,
                indicators=inds,
                sentiment=sentiment,
                threshold=0.02
            )

            # --- Persist query ---
            q = Query(
                ticker        = ticker,
                start_date    = datetime.fromisoformat(start_date).date(),
                end_date      = datetime.fromisoformat(end_date).date(),
                weeks         = weeks,
                recommendation= signal
            )
            db.session.add(q)
            db.session.commit()

            # --- Chart Data ---
            proc_csv = os.path.join(ROOT_DIR, 'data', 'processed', f'{ticker}_features.csv')
            hist_df  = pd.read_csv(proc_csv, index_col='Date', parse_dates=True)
            mask     = (
                (hist_df.index.date >= datetime.fromisoformat(start_date).date()) &
                (hist_df.index.date <= datetime.fromisoformat(end_date).date())
            )
            history = hist_df.loc[mask]

            hist_x = history.index.strftime('%Y-%m-%d').tolist()
            hist_y = history['Close'].tolist()
            fc_x   = pd.bdate_range(
                        start=history.index[-1] + timedelta(days=1),
                        periods=len(preds)
                     ).strftime('%Y-%m-%d').tolist()
            fc_y   = preds

            return render_template(
                'result.html',
                ticker     = ticker,
                weeks      = weeks,
                indicators = inds,
                sentiment  = sentiment,
                advice     = advice_text,
                hist_x     = hist_x,
                hist_y     = hist_y,
                fc_x       = fc_x,
                fc_y       = fc_y
            )

        except Exception as e:
            current_app.logger.exception("Error in / POST")
            flash(f"Unexpected error: {e}", 'danger')
            return redirect(url_for('index'))

    # GET → show form
    return render_template(
        'index.html',
        tickers      = AVAILABLE_TICKERS,
        default_start= default_start,
        default_end  = default_end
    )