# Vercel entry point: every request is rewritten here (see vercel.json) and
# served by the Flask app.
from src.webapp.app import app  # noqa: F401
