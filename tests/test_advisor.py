from src.advisor import get_signal


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
