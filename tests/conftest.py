import os

# Keep the test run off the real database; must be set before the app is imported.
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ.pop("NEWSAPI_KEY", None)
