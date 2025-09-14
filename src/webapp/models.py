from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Query(db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    ticker       = db.Column(db.String(10), nullable=False)
    start_date   = db.Column(db.Date, nullable=False)
    end_date     = db.Column(db.Date, nullable=False)
    weeks        = db.Column(db.Integer, nullable=False)
    recommendation = db.Column(db.String(10), nullable=False)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)
