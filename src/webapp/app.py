# src/webapp/app.py
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from .models import db

app = Flask(
    __name__,
    template_folder='templates',
    static_folder='static'
)

# ← Add this:
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret')  # replace 'dev-secret' with a strong key in prod

app.config['DEBUG'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///advisor.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
with app.app_context():
    db.create_all()

from . import routes

if __name__ == '__main__':
    app.run(debug=True)
