from datetime import date, timedelta
from typing import Optional

import pandas as pd

_sia = None

# VADER is tuned for social media; these words carry clear meaning in market
# headlines but are missing from (or neutral in) its stock lexicon.
FINANCE_LEXICON = {
    "soar": 2.5, "soars": 2.5, "soared": 2.5, "surge": 2.3, "surges": 2.3, "surged": 2.3,
    "rally": 2.0, "rallies": 2.0, "rallied": 2.0, "jump": 1.6, "jumps": 1.6, "jumped": 1.6,
    "gain": 1.5, "gains": 1.5, "climb": 1.4, "climbs": 1.4, "rebound": 1.5, "rebounds": 1.5,
    "beat": 1.8, "beats": 1.8, "upgrade": 2.0, "upgrades": 2.0, "upgraded": 2.0,
    "outperform": 1.8, "bullish": 2.2, "record": 1.2, "boost": 1.6, "boosts": 1.6,
    "plunge": -2.6, "plunges": -2.6, "plunged": -2.6, "slump": -2.2, "slumps": -2.2,
    "tumble": -2.2, "tumbles": -2.2, "tumbled": -2.2, "sink": -1.8, "sinks": -1.8,
    "drop": -1.5, "drops": -1.5, "fall": -1.5, "falls": -1.5, "slide": -1.5, "slides": -1.5,
    "crash": -3.0, "crashes": -3.0, "miss": -1.8, "misses": -1.8, "missed": -1.8,
    "downgrade": -2.0, "downgrades": -2.0, "downgraded": -2.0, "underperform": -1.8,
    "bearish": -2.2, "selloff": -2.2, "sell-off": -2.2, "lawsuit": -1.6, "probe": -1.4,
    "layoffs": -1.8, "recall": -1.5, "warns": -1.6, "cut": -1.0, "cuts": -1.0,
}


def _get_sia():
    # Built on first use: the VADER lexicon is only needed when scoring news.
    global _sia
    if _sia is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _sia = SentimentIntensityAnalyzer()
        _sia.lexicon.update(FINANCE_LEXICON)
    return _sia


def score_headline(text: str) -> float:
    """VADER compound score in [-1, 1] for one headline."""
    return _get_sia().polarity_scores(text)["compound"]


def get_signal(current: float, predicted: float, threshold: float = 0.02):
    # Ensure scalar floats
    current = float(current)
    predicted = float(predicted)
    pct = (predicted - current) / current

    if pct > threshold:
        return "Buy", pct
    elif pct < -threshold:
        return "Sell", pct
    else:
        return "Hold", pct


def format_advice(current: float,
                  predicted: float,
                  pct: float,
                  signal: str,
                  indicators: dict,
                  sentiment: Optional[float],
                  threshold: float = 0.02) -> str:
    """
    Generate user-friendly advice with reasoning from indicators and sentiment.
    Pass sentiment=None when no news data is available.
    """
    reasoning = []

    # Trend logic
    if indicators['MA50'] > indicators['MA200']:
        reasoning.append("the short-term trend (MA50) is above the long-term trend (MA200), showing bullish momentum")
    else:
        reasoning.append("the short-term trend (MA50) is below the long-term trend (MA200), suggesting weakness")

    # RSI logic
    if indicators['RSI'] > 70:
        reasoning.append("RSI is above 70, which means the stock may be overbought")
    elif indicators['RSI'] < 30:
        reasoning.append("RSI is below 30, which means the stock may be oversold")
    else:
        reasoning.append("RSI is in the neutral zone")

    # MACD logic
    if indicators['MACD'] > 0:
        reasoning.append("MACD is positive, supporting upward momentum")
    else:
        reasoning.append("MACD is negative, signaling possible downward pressure")

    # Sentiment logic
    if sentiment is not None:
        if sentiment > 0.2:
            reasoning.append("news sentiment is generally positive")
        elif sentiment < -0.2:
            reasoning.append("news sentiment is mostly negative")
        else:
            reasoning.append("news sentiment is neutral")

    return (
        f"The analysis suggests a {signal}.\n\n"
        f"The model predicts the price to move from ${current:.2f} "
        f"to ${predicted:.2f}, a change of {pct*100:.2f}%.\n\n"
        f"This decision is based on indicators: {', '.join(reasoning)}."
    )


def compute_last_indicators(df: pd.DataFrame) -> dict:
    """
    Always return Python floats for safety.
    """
    return {
        "Close": float(df["Close"].iloc[-1]),
        "MA50": float(df["MA50"].iloc[-1]),
        "MA200": float(df["MA200"].iloc[-1]),
        "MACD": float(df["MACD"].iloc[-1]),
        "RSI": float(df["RSI"].iloc[-1]),
    }


def fetch_news_sentiment(newsapi, ticker: str, days: int = 7) -> Optional[float]:
    """
    Mean VADER compound score of recent headlines mentioning the ticker.
    Returns None when there is no NewsAPI client or no headlines.
    """
    if newsapi is None:
        return None

    articles = newsapi.get_everything(
        q=ticker,
        from_param=(date.today() - timedelta(days=days)).isoformat(),
        language='en',
        sort_by='relevancy',
        page_size=20
    )['articles']

    titles = [a['title'] for a in articles if a.get('title')]
    if not titles:
        return None

    return sum(score_headline(t) for t in titles) / len(titles)
