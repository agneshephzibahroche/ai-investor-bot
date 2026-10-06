from src.advisor import fetch_news_sentiment, format_advice, get_signal


def test_get_signal_buy():
    signal, pct = get_signal(current=100.0, predicted=105.0, threshold=0.02)
    assert signal == "Buy"
    assert pct > 0.02


def test_get_signal_sell():
    signal, pct = get_signal(current=100.0, predicted=95.0, threshold=0.02)
    assert signal == "Sell"
    assert pct < -0.02


def test_get_signal_hold():
    signal, pct = get_signal(current=100.0, predicted=100.5, threshold=0.02)
    assert signal == "Hold"
    assert abs(pct) <= 0.02


INDICATORS = {"Close": 100.0, "MA50": 105.0, "MA200": 100.0, "MACD": 1.2, "RSI": 75.0}


def test_format_advice_explains_signal():
    text = format_advice(100.0, 105.0, 0.05, "Buy", INDICATORS, sentiment=0.5)
    assert "Buy" in text
    assert "$100.00" in text and "$105.00" in text and "5.00%" in text
    assert "bullish momentum" in text
    assert "overbought" in text
    assert "news sentiment is generally positive" in text


def test_format_advice_omits_sentiment_when_unavailable():
    text = format_advice(100.0, 105.0, 0.05, "Buy", INDICATORS, sentiment=None)
    assert "sentiment" not in text


def test_fetch_news_sentiment_without_client():
    assert fetch_news_sentiment(None, "AAPL") is None
