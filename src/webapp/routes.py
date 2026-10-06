# src/webapp/routes.py

import os
import pandas as pd
from datetime import date, datetime, timedelta
from flask import render_template, request, flash, redirect, url_for, current_app
from newsapi import NewsApiClient

from src.webapp.app import app
from src.webapp.models import db, Query
from src.inference import MODEL_DIR, load_resources, fetch_and_prepare, recursive_forecast
from src.advisor   import get_signal, format_advice, compute_last_indicators, fetch_news_sentiment

MAX_WEEKS        = 12   # each week is 5 sequential model calls
SIGNAL_THRESHOLD = 0.02


def get_available_tickers():
    """Return tickers that have a model and both scalers in models/."""
    if not os.path.isdir(MODEL_DIR):
        return []
    files = set(os.listdir(MODEL_DIR))
    tickers = [
        fn[len("lstm_"):-len(".h5")]
        for fn in files
        if fn.startswith("lstm_") and fn.endswith(".h5")
    ]
    return sorted(
        t for t in tickers
        if f"scaler_{t}.save" in files and f"scaler_close_{t}.save" in files
    )


# Init NewsAPI (optional)
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', '')
newsapi     = NewsApiClient(api_key=NEWSAPI_KEY) if NEWSAPI_KEY else None


def parse_form(form, tickers):
    """Validate the query form. Returns (ticker, weeks, start, end) or raises ValueError."""
    ticker = form.get('ticker', '').strip().upper()
    if ticker not in tickers:
        raise ValueError(f"No model available for “{ticker}”. Please select from the dropdown.")

    try:
        weeks = int(form.get('weeks', 1))
    except ValueError:
        raise ValueError("Horizon must be a whole number of weeks.")
    if not 1 <= weeks <= MAX_WEEKS:
        raise ValueError(f"Horizon must be between 1 and {MAX_WEEKS} weeks.")

    try:
        start = date.fromisoformat(form.get('start_date', ''))
        end   = date.fromisoformat(form.get('end_date', ''))
    except ValueError:
        raise ValueError("Please enter valid start and end dates.")
    if start >= end:
        raise ValueError("Start date must be before end date.")

    return ticker, weeks, start, end


@app.route('/health')
def health():
    return {'status': 'ok'}, 200


@app.route('/', methods=['GET', 'POST'])
def index():
    today   = datetime.today().date()
    tickers = get_available_tickers()

    if request.method == 'POST':
        try:
            ticker, weeks, start_date, end_date = parse_form(request.form, tickers)
        except ValueError as e:
            flash(str(e), 'danger')
            return redirect(url_for('index'))

        try:
            # --- Load scalers + model ---
            scaler_all, scaler_close, model = load_resources(ticker)

            # --- Prepare features ---
            try:
                features, full_df = fetch_and_prepare(ticker)
            except ValueError as e:
                flash(str(e), 'danger')
                return redirect(url_for('index'))

            # --- Forecast ---
            days  = weeks * 5
            preds = recursive_forecast(features, scaler_all, scaler_close, model, days)

            # --- Signals (force scalars) ---
            current_close = float(full_df['Close'].iloc[-1])
            forecast_end  = float(preds[-1])
            signal, pct_change = get_signal(current_close, forecast_end, SIGNAL_THRESHOLD)

            # --- Indicators ---
            inds = compute_last_indicators(full_df)

            # --- News Sentiment (optional; never fails the request) ---
            try:
                sentiment = fetch_news_sentiment(newsapi, ticker)
            except Exception:
                current_app.logger.exception("News sentiment lookup failed")
                sentiment = None

            # --- Advice with reasoning ---
            advice_text = format_advice(
                current=current_close,
                predicted=forecast_end,
                pct=pct_change,
                signal=signal,
                indicators=inds,
                sentiment=sentiment,
                threshold=SIGNAL_THRESHOLD
            )

            # --- Persist query ---
            q = Query(
                ticker        = ticker,
                start_date    = start_date,
                end_date      = end_date,
                weeks         = weeks,
                recommendation= signal
            )
            db.session.add(q)
            db.session.commit()

            # --- Chart Data ---
            mask    = (full_df.index.date >= start_date) & (full_df.index.date <= end_date)
            history = full_df.loc[mask]
            if history.empty:
                history = full_df.tail(90)

            hist_x = history.index.strftime('%Y-%m-%d').tolist()
            hist_y = history['Close'].tolist()
            # The forecast always continues from the latest available close
            fc_x   = pd.bdate_range(
                        start=full_df.index[-1] + timedelta(days=1),
                        periods=len(preds)
                     ).strftime('%Y-%m-%d').tolist()
            fc_y   = preds

            return render_template(
                'result.html',
                ticker     = ticker,
                weeks      = weeks,
                signal     = signal,
                indicators = inds,
                sentiment  = sentiment,
                advice     = advice_text,
                hist_x     = hist_x,
                hist_y     = hist_y,
                fc_x       = fc_x,
                fc_y       = fc_y
            )

        except Exception:
            db.session.rollback()
            current_app.logger.exception("Error in / POST")
            flash("Something went wrong while generating the forecast. Please try again.", 'danger')
            return redirect(url_for('index'))

    # GET → show form
    return render_template(
        'index.html',
        tickers      = tickers,
        max_weeks    = MAX_WEEKS,
        default_start= (today - timedelta(days=90)).isoformat(),
        default_end  = today.isoformat()
    )
