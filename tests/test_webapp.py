# tests/test_webapp.py

import pytest
from src.webapp.app import app

@pytest.fixture
def client():
    # Enable testing mode
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    """GET / should return the homepage."""
    resp = client.get('/')
    assert resp.status_code == 200
    assert b'Financial Advisor Bot' in resp.data

def test_advice_route(client):
    """GET /advice should return a recommendation page."""
    resp = client.get('/advice')
    assert resp.status_code == 200
    # should contain the “Recommendation:” header
    assert b'Recommendation:' in resp.data
