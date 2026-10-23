import pytest

from src import inference
from src.market_data import SymbolNotFound
from src.webapp import routes
from src.webapp.app import app
from src.webapp.models import Forecast
from tests.test_analytics import long_prices
from tests.test_inference import ConstantModel

PRICES = long_prices()
QUOTE = {"symbol": "TEST", "name": "Test Corp", "price": 101.0, "prev_close": 100.0, "change": 1.0,
         "change_pct": 1.0, "currency": "USD", "type": "EQUITY", "spark": [100.0, 101.0]}
META = {"target_scale": 10.0, "tickers": ["TEST"], "trained_on": "2026-01-01",
        "test_mape": 0.012, "naive_mape": 0.012, "test_direction_acc": 0.52, "test_samples": 100.0}


def fake_history(symbol, range_="5y"):
    if symbol == "NOPE":
        raise SymbolNotFound("We couldn't find a ticker called NOPE.")
    return PRICES, {"currency": "USD"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(routes, "get_history", fake_history)
    monkeypatch.setattr(inference, "get_history", fake_history)
    monkeypatch.setattr(routes, "get_quotes", lambda syms: [dict(QUOTE, symbol=s) for s in syms])
    monkeypatch.setattr(routes, "get_news", lambda s: [{"title": "Test Corp shares soar on record profit", "link": "#"}])
    monkeypatch.setattr(routes, "get_profile", lambda s: {"sector": "Tech"})
    monkeypatch.setattr(routes, "search", lambda q: [{"symbol": "TEST", "name": "Test Corp"}])
    monkeypatch.setattr(routes, "available_models", lambda: ["universal"])
    monkeypatch.setattr(routes, "model_for", lambda s: "universal")
    # +1% a day: comfortably a Buy
    monkeypatch.setattr(routes, "load_resources", lambda name: (ConstantModel(0.1), META))
    app.config["TESTING"] = True
    return app.test_client()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200 and resp.get_json()["status"] == "ok"


@pytest.mark.parametrize("path", ["/", "/stock/TEST", "/compare"])
def test_pages_render(client, path):
    resp = client.get(path)
    assert resp.status_code == 200
    assert b"AI Investor" in resp.data


def test_search_param_redirects_to_stock_page(client):
    resp = client.get("/?ticker=aapl")
    assert resp.status_code == 302 and resp.headers["Location"].endswith("/stock/AAPL")
    assert client.get("/stock/msft").headers["Location"].endswith("/stock/MSFT")


def test_analysis(client):
    data = client.get("/api/analysis/TEST").get_json()
    assert data["quote"]["name"] == "Test Corp" and data["model"] == "universal"
    assert len(data["history"]["dates"]) == len(PRICES)
    assert -100 <= data["scorecard"]["score"] <= 100
    assert data["risk"]["beta"] == pytest.approx(1.0)


def test_forecast_returns_signal_and_saves_it(client):
    resp = client.get("/api/forecast/TEST?weeks=2")
    assert resp.status_code == 200
    assert "s-maxage" in resp.headers["Cache-Control"]
    data = resp.get_json()
    assert data["signal"] == "Buy" and data["model"] == "lstm"
    assert len(data["dates"]) == len(data["lstm"]["path"]) == 10
    assert len(data["monte_carlo"]["p50"]) == 10
    assert "The analysis suggests a Buy." in data["advice"]
    assert data["sentiment"] > 0  # "soar" and "record" are bullish words
    with app.app_context():
        saved = Forecast.query.order_by(Forecast.id.desc()).first()
        assert (saved.ticker, saved.weeks, saved.recommendation) == ("TEST", 2, "Buy")


def test_forecast_without_model_uses_monte_carlo(client, monkeypatch):
    monkeypatch.setattr(routes, "model_for", lambda s: None)
    data = client.get("/api/forecast/TEST?weeks=1").get_json()
    assert data["model"] == "monte_carlo" and data["lstm"] is None
    assert data["target"] == data["monte_carlo"]["p50"][-1]


@pytest.mark.parametrize("weeks, message", [("abc", "whole number"), ("0", "between 1 and"), ("999", "between 1 and")])
def test_forecast_rejects_bad_horizon(client, weeks, message):
    resp = client.get(f"/api/forecast/TEST?weeks={weeks}")
    assert resp.status_code == 400 and message in resp.get_json()["error"]


def test_bad_and_unknown_symbols(client):
    assert client.get("/api/analysis/bad%20sym!").status_code == 400
    resp = client.get("/api/analysis/NOPE")
    assert resp.status_code == 404 and "NOPE" in resp.get_json()["error"]


def test_backtest_news_compare_market(client):
    bt = client.get("/api/backtest/TEST?years=2").get_json()
    assert bt["strategies"][0]["key"] == "buy_hold"
    news = client.get("/api/news/TEST").get_json()
    assert news["items"][0]["sentiment"] > 0
    cmp_ = client.get("/api/compare?symbols=TEST,SPY").get_json()
    assert set(cmp_["series"]) == {"TEST", "SPY"}
    assert client.get("/api/compare?symbols=TEST").status_code == 400
    market = client.get("/api/market").get_json()
    assert market["overview"] and market["sectors"] and market["models"][0]["name"] == "universal"


def test_internal_errors_are_not_shown_to_the_user(client, monkeypatch):
    def boom(symbol, range_="5y"):
        raise RuntimeError("secret/path/to/model.npz")

    monkeypatch.setattr(routes, "get_history", boom)
    monkeypatch.setitem(app.config, "PROPAGATE_EXCEPTIONS", False)
    resp = client.get("/api/analysis/TEST")
    assert resp.status_code == 500
    assert "secret" not in resp.get_data(as_text=True)


def test_portfolio(client):
    assert client.get("/portfolio").status_code == 200
    data = client.get("/api/portfolio?holdings=TEST:10,SPY:2.5,TEST:5").get_json()
    assert set(data["weights"]) == {"TEST", "SPY"}  # duplicate TEST lines are merged
    assert data["value"][-1] == pytest.approx(17.5 * PRICES["Close"].iloc[-1], abs=0.05)
    assert data["stats"]["benchmark_return"] is not None


@pytest.mark.parametrize("holdings, message", [
    ("", "at least one"), ("TEST:abc", "must be a number"), ("TEST:-3", "must be positive"),
])
def test_portfolio_rejects_bad_holdings(client, holdings, message):
    resp = client.get(f"/api/portfolio?holdings={holdings}")
    assert resp.status_code == 400 and message in resp.get_json()["error"]


def test_share_preview_tags(client):
    home = client.get("/").get_data(as_text=True)
    assert 'property="og:image" content="http://localhost/static/og.png"' in home
    assert "smarter stock research" in home
    page = client.get("/stock/TEST").get_data(as_text=True)
    assert 'content="TEST: AI forecast, chart &amp; analysis"' in page
