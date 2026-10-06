# src/webapp/app.py
import os
from flask import Flask
from .models import db

app = Flask(
    __name__,
    template_folder='templates',
    static_folder='static'
)

app.secret_key = os.environ['SECRET_KEY'] if os.getenv('FLASK_ENV') == 'production' else os.getenv('SECRET_KEY', 'dev-secret')

app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///advisor.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Hugging Face shows the app inside an iframe, which needs cross-site session cookies
if os.getenv('SPACE_ID'):
    app.config['SESSION_COOKIE_SAMESITE'] = 'None'
    app.config['SESSION_COOKIE_SECURE'] = True

db.init_app(app)
with app.app_context():
    db.create_all()

from . import routes

if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'], host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
