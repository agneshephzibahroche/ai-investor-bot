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

# Vercel's filesystem is read-only apart from /tmp, so the default SQLite file
# lives there (and resets on cold starts). Set DATABASE_URL to keep history.
default_db = 'sqlite:////tmp/advisor.db' if os.getenv('VERCEL') else 'sqlite:///advisor.db'

app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', default_db)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.json.sort_keys = False  # keep returns, stats etc. in the order they're built

# Hugging Face shows the app inside an iframe, which needs cross-site session cookies
if os.getenv('SPACE_ID'):
    app.config['SESSION_COOKIE_SAMESITE'] = 'None'
    app.config['SESSION_COOKIE_SECURE'] = True

# Behind Vercel's proxy: trust its forwarded scheme/host so absolute URLs
# (the social preview image) use https and the public domain
if os.getenv('VERCEL'):
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

db.init_app(app)
with app.app_context():
    db.create_all()

from . import routes  # noqa: E402,F401

if __name__ == '__main__':
    # Local dev only; set HOST=0.0.0.0 to expose it on your network. Production uses gunicorn or Vercel.
    app.run(debug=app.config['DEBUG'], host=os.getenv('HOST', '127.0.0.1'), port=int(os.getenv('PORT', 5000)))
