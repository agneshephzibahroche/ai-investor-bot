from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone

db = SQLAlchemy()


class Forecast(db.Model):
    """One forecast a visitor asked for; feeds the "recently analyzed" list."""
    id             = db.Column(db.Integer, primary_key=True)
    ticker         = db.Column(db.String(20), nullable=False, index=True)
    weeks          = db.Column(db.Integer, nullable=False)
    recommendation = db.Column(db.String(10), nullable=False)
    model          = db.Column(db.String(20), nullable=False)
    pct_change     = db.Column(db.Float)
    created_at     = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)
