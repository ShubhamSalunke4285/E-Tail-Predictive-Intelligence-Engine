"""WSGI entry point for production servers (gunicorn / Render / Railway).

    gunicorn wsgi:app --bind 0.0.0.0:$PORT
"""
from src.app.app import app

if __name__ == "__main__":
    app.run()
