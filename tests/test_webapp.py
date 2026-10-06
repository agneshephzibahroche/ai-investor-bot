import numpy as np
import pytest

from src.feature_engineering import FEATURES, compute_indicators
from src.inference import SEQ_LENGTH
from src.webapp import routes
from src.webapp.app import app
from src.webapp.models import Query
from tests.test_feature_engineering import make_prices
from tests.test_inference import ConstantModel, IdentityScaler


@pytest.fixture
def client(monkeypatch):
    df = compute_indicators(make_prices()).dropna()[FEATURES]

    monkeypatch.setattr(routes, "get_available_tickers", lambda: ["TEST"])
    monkeypatch.setattr(
        routes, "load_resources",
        lambda ticker: (IdentityScaler(), IdentityScaler(), ConstantModel(1e6)),
    )
    monkeypatch.setattr(
        routes, "fetch_and_prepare", lambda ticker: (df.tail(SEQ_LENGTH).values, df)
    )
    app.config["TESTING"] = True
    return app.test_client()


def valid_form(**overrides):
    form = {"ticker": "TEST", "weeks": "2", "start_date": "2023-06-01", "end_date": "2024-03-01"}
    form.update(overrides)
    return form


def flashed(client):
    with client.session_transaction() as session:
        return [msg for _, msg in session.get("_flashes", [])]


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_index_lists_tickers(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b'<option value="TEST">' in resp.data


def test_forecast_renders_result_and_saves_query(client):
    resp = client.post("/", data=valid_form())
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "Forecast for TEST over 2 Weeks" in body
    assert "The analysis suggests a Buy." in body
    assert "alert-success" in body

    with app.app_context():
        saved = Query.query.order_by(Query.id.desc()).first()
        assert (saved.ticker, saved.weeks, saved.recommendation) == ("TEST", 2, "Buy")


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"ticker": "NOPE"}, "No model available"),
        ({"weeks": "abc"}, "whole number"),
        ({"weeks": "0"}, "between 1 and"),
        ({"weeks": "100000"}, "between 1 and"),
        ({"start_date": "not-a-date"}, "valid start and end dates"),
        ({"start_date": "2024-03-01", "end_date": "2023-06-01"}, "before end date"),
    ],
)
def test_invalid_form_is_rejected(client, overrides, message):
    resp = client.post("/", data=valid_form(**overrides))
    assert resp.status_code == 302
    assert any(message in msg for msg in flashed(client))


def test_internal_errors_are_not_shown_to_the_user(client, monkeypatch):
    def boom(ticker):
        raise RuntimeError("secret/path/to/model.h5")

    monkeypatch.setattr(routes, "load_resources", boom)
    resp = client.post("/", data=valid_form())
    assert resp.status_code == 302
    messages = flashed(client)
    assert messages and not any("secret" in msg for msg in messages)
