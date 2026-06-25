"""Single entrypoint for the ShopSphere predictive e-commerce platform.

Usage:
    python run.py setup        # seed the app DB from the Kaggle dataset
    python run.py pipeline      # run ETL + CF + ARIMA + inventory (retrain)
    python run.py web           # launch the storefront + admin web app
    python run.py all           # setup (if needed) -> pipeline -> web
"""
import sys


def _setup():
    from src import setup_database
    setup_database.build()


def _pipeline():
    from src.pipeline import run_pipeline
    run_pipeline()


def _web():
    from src.app.app import app
    print("ShopSphere running at  http://127.0.0.1:5000")
    print("  storefront -> http://127.0.0.1:5000/")
    print("  admin      -> http://127.0.0.1:5000/admin")
    app.run(debug=False, port=5000)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "setup":
        _setup()
    elif cmd == "pipeline":
        _pipeline()
    elif cmd == "web":
        _web()
    elif cmd == "all":
        _pipeline()       # auto-seeds the DB on first run
        _web()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
