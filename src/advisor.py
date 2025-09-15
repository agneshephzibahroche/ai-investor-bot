import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()


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
                  sentiment: float,
                  threshold: float = 0.02) -> str:
    """
    Generate user-friendly advice with reasoning from indicators and sentiment.
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
    if sentiment > 0.2:
        reasoning.append("news sentiment is generally positive")
    elif sentiment < -0.2:
        reasoning.append("news sentiment is mostly negative")
    else:
        reasoning.append("news sentiment is neutral")

    return (
        f"The analysis suggests a **{signal}**.\n\n"
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


def fetch_news_sentiment(ticker: str) -> float:
    """
    Very simple sentiment function (replace with NewsAPI if available).
    """
    headlines = [f"Company {ticker} sees record revenue growth"]
    text = " ".join(headlines)
    return sia.polarity_scores(text)["compound"]