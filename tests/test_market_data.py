import pytest

from src import market_data


CHART = {
    "timestamp": [1704205800, 1704292200, 1704378600],
    "indicators": {
        "quote": [{"open": [10.0, 11.0, None], "high": [12.0, 12.0, None], "low": [9.0, 10.0, None],
                   "close": [11.0, 11.5, None], "volume": [100, 200, None]}],
        "adjclose": [{"adjclose": [5.5, 5.75, None]}],
    },
}


def test_frame_from_chart_adjusts_and_drops_gaps():
    df = market_data._frame_from_chart(CHART)
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(df) == 2  # the null bar is dropped
    assert df["Close"].tolist() == [5.5, 5.75]
    assert df["Open"].iloc[0] == pytest.approx(5.0)  # scaled by adjclose / close
    assert df.index[0].hour == 0


def test_get_history_reports_unknown_symbols(monkeypatch):
    market_data.get_history.cache_clear()
    monkeypatch.delenv("TWELVEDATA_API_KEY", raising=False)
    monkeypatch.setattr(market_data, "_yahoo", lambda path, params: {"chart": {"result": None}})
    with pytest.raises(market_data.SymbolNotFound):
        market_data.get_history("NOPE")


def test_get_history_reports_outages(monkeypatch):
    market_data.get_history.cache_clear()
    monkeypatch.delenv("TWELVEDATA_API_KEY", raising=False)

    def down(path, params):
        raise market_data.MarketDataError("timeout")

    monkeypatch.setattr(market_data, "_yahoo", down)
    with pytest.raises(market_data.MarketDataError) as err:
        market_data.get_history("AAPL")
    assert not isinstance(err.value, market_data.SymbolNotFound)


def test_quote_change_uses_previous_close():
    q = market_data._quote_from_meta({"symbol": "X", "regularMarketPrice": 110.0, "chartPreviousClose": 100.0},
                                     [100.0, None, 105.0, 110.0])
    assert q["change"] == 10.0 and q["change_pct"] == pytest.approx(10.0)
    assert q["spark"] == [100.0, 105.0, 110.0]


def test_search_filters_to_tradable_types(monkeypatch):
    market_data.search.cache_clear()
    monkeypatch.setattr(market_data, "_yahoo", lambda path, params: {"quotes": [
        {"symbol": "NVDA", "shortname": "NVIDIA", "quoteType": "EQUITY", "exchDisp": "NASDAQ"},
        {"symbol": "NVDA240119C", "quoteType": "OPTION"},
    ]})
    assert [r["symbol"] for r in market_data.search("nvda")] == ["NVDA"]


def test_ttl_cache_expires(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(market_data.time, "monotonic", lambda: clock[0])
    calls = []

    @market_data.ttl_cache(10)
    def f(x):
        calls.append(x)
        return x * 2

    assert f(2) == 4 and f(2) == 4 and calls == [2]
    clock[0] = 11
    f(2)
    assert calls == [2, 2]
