"""Flask e-commerce application: shopper storefront + admin analytics.

Shopper view  -> browse / search a catalog, every click is logged to the ETL
                 clickstream, and recommendations update from that behaviour.
Admin view    -> ARIMA sales forecasts, inventory reorder decisions and a
                 slow-mover / clearance engine, plus an on-demand model retrain.
"""
from __future__ import annotations

import os

import pandas as pd
from flask import (Flask, jsonify, redirect, render_template, request,
                   session, url_for)

import config
from src import db

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "etail-predictive-intelligence-demo")

CATEGORY_EMOJI = {
    "Seasonal": "🎄", "Kitchen & Dining": "🍽️", "Home Decor": "🏠",
    "Bags & Travel": "👜", "Stationery & Gift": "🎁", "Garden & Outdoor": "🌿",
    "Toys & Games": "🧸", "Jewellery": "💍", "Lighting": "💡", "General": "🛍️",
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def current_user():
    uid = session.get("user_id")
    if uid is None:
        return None
    return {"id": uid, "name": session.get("user_name", f"Customer {uid}")}


def log_event(user_id, product_id, event_type, qty=1):
    """Append a live event to the ETL clickstream."""
    if user_id is None:
        return
    db.execute(
        "INSERT INTO interaction_logs "
        "(user_id, product_id, event_type, qty, timestamp, source) "
        "VALUES (:u, :p, :e, :q, :t, 'live')",
        {"u": int(user_id), "p": int(product_id), "e": event_type,
         "q": int(qty), "t": pd.Timestamp.now().isoformat(timespec="seconds")},
    )


def emoji(cat):
    return CATEGORY_EMOJI.get(cat, "🛍️")


def products_to_records(df):
    df = df.copy()
    df["emoji"] = df["category"].map(emoji)
    return df.to_dict(orient="records")


def get_categories():
    df = db.query(
        "SELECT category, COUNT(*) n FROM products GROUP BY category ORDER BY n DESC"
    )
    df["emoji"] = df["category"].map(emoji)
    return df.to_dict(orient="records")


def live_recommendations(user_id, limit=8):
    """Fresh, session-based recs built from the user's most recent live clicks
    via the precomputed item-similarity map (no retrain needed)."""
    if user_id is None or not db.table_exists("item_similar"):
        return []
    recent = db.query(
        "SELECT DISTINCT product_id FROM interaction_logs "
        "WHERE user_id = :u AND source = 'live' "
        "ORDER BY timestamp DESC LIMIT 10", {"u": int(user_id)})
    if recent.empty:
        return []
    ids = ",".join(str(int(x)) for x in recent["product_id"])
    df = db.query(
        f"SELECT s.similar_id AS product_id, SUM(s.score) AS score "
        f"FROM item_similar s WHERE s.product_id IN ({ids}) "
        f"AND s.similar_id NOT IN ({ids}) "
        f"GROUP BY s.similar_id ORDER BY score DESC LIMIT {limit}")
    if df.empty:
        return []
    prod = db.query(
        f"SELECT product_id, name, category, price, stock_qty "
        f"FROM products WHERE product_id IN "
        f"({','.join(str(int(x)) for x in df['product_id'])})")
    out = df.merge(prod, on="product_id")
    return products_to_records(out)


def personalized_recommendations(user_id, limit=8):
    """Precomputed CF recommendations (refreshed on retrain)."""
    if user_id is None or not db.table_exists("recommendations"):
        return []
    df = db.query(
        "SELECT p.product_id, p.name, p.category, p.price, p.stock_qty, r.score "
        "FROM recommendations r JOIN products p ON p.product_id = r.product_id "
        "WHERE r.user_id = :u ORDER BY r.rank LIMIT :n",
        {"u": int(user_id), "n": limit})
    return products_to_records(df)


def trending(limit=8):
    df = db.query(
        "SELECT product_id, name, category, price, stock_qty "
        "FROM products ORDER BY orders DESC LIMIT :n", {"n": limit})
    return products_to_records(df)


# ----------------------------------------------------------------------------
# Storefront
# ----------------------------------------------------------------------------
@app.route("/")
def home():
    user = current_user()
    uid = user["id"] if user else None
    return render_template(
        "store.html", user=user, categories=get_categories(),
        trending=trending(8),
        live_recs=live_recommendations(uid, 8),
        for_you=personalized_recommendations(uid, 8),
    )


@app.route("/browse")
def browse():
    user = current_user()
    q = request.args.get("q", "").strip()
    category = request.args.get("category", "").strip()
    sort = request.args.get("sort", "popular")

    if q and user:
        log_event(user["id"], _pick_search_target(q), "search")

    sql = ("SELECT product_id, name, category, price, stock_qty, orders "
           "FROM products WHERE 1=1")
    params = {}
    if q:
        sql += " AND UPPER(name) LIKE :q"
        params["q"] = f"%{q.upper()}%"
    if category:
        sql += " AND category = :c"
        params["c"] = category
    order = {"price_low": "price ASC", "price_high": "price DESC",
             "popular": "orders DESC"}.get(sort, "orders DESC")
    sql += f" ORDER BY {order} LIMIT 60"

    products = db.query(sql, params)
    return render_template(
        "browse.html", user=user, products=products_to_records(products),
        categories=get_categories(), q=q, category=category, sort=sort,
        count=len(products))


def _pick_search_target(q):
    """Attribute a search event to the top matching product (for the ETL log)."""
    hit = db.query("SELECT product_id FROM products WHERE UPPER(name) LIKE :q "
                   "ORDER BY orders DESC LIMIT 1", {"q": f"%{q.upper()}%"})
    return int(hit["product_id"].iloc[0]) if not hit.empty else 0


@app.route("/product/<int:pid>")
def product(pid):
    user = current_user()
    row = db.query("SELECT * FROM products WHERE product_id = :p", {"p": pid})
    if row.empty:
        return redirect(url_for("home"))
    p = row.iloc[0].to_dict()
    p["emoji"] = emoji(p["category"])

    if user:
        log_event(user["id"], pid, "view")        # <-- ETL clickstream capture

    similar = db.query(
        "SELECT p.product_id, p.name, p.category, p.price, p.stock_qty "
        "FROM item_similar s JOIN products p ON p.product_id = s.similar_id "
        "WHERE s.product_id = :p ORDER BY s.score DESC LIMIT 6", {"p": pid}) \
        if db.table_exists("item_similar") else pd.DataFrame()

    return render_template(
        "product.html", user=user, p=p,
        similar=products_to_records(similar) if not similar.empty else [],
        for_you=personalized_recommendations(user["id"] if user else None, 4))


@app.route("/cart/add/<int:pid>", methods=["POST"])
def cart_add(pid):
    user = current_user()
    cart = session.get("cart", [])
    cart.append(pid)
    session["cart"] = cart
    if user:
        log_event(user["id"], pid, "add_to_cart")
    return jsonify({"ok": True, "cart_count": len(cart)})


@app.route("/buy/<int:pid>", methods=["POST"])
def buy(pid):
    user = current_user()
    if not user:
        return jsonify({"ok": False, "need_login": True})
    log_event(user["id"], pid, "purchase", qty=1)
    db.execute("UPDATE products SET stock_qty = MAX(stock_qty - 1, 0) "
               "WHERE product_id = :p", {"p": pid})
    return jsonify({"ok": True})


@app.route("/cart")
def cart():
    user = current_user()
    ids = session.get("cart", [])
    items = []
    if ids:
        idlist = ",".join(str(int(i)) for i in set(ids))
        df = db.query(f"SELECT product_id, name, category, price, stock_qty "
                      f"FROM products WHERE product_id IN ({idlist})")
        counts = pd.Series(ids).value_counts()
        df["qty"] = df["product_id"].map(counts).fillna(1).astype(int)
        df["line_total"] = (df["price"] * df["qty"]).round(2)
        items = products_to_records(df)
    total = round(sum(i["line_total"] for i in items), 2)
    return render_template("cart.html", user=user, items=items, total=total)


@app.route("/checkout", methods=["POST"])
def checkout():
    user = current_user()
    ids = session.get("cart", [])
    if user:
        for pid in ids:
            log_event(user["id"], pid, "purchase")
            db.execute("UPDATE products SET stock_qty = MAX(stock_qty - 1, 0) "
                       "WHERE product_id = :p", {"p": int(pid)})
    session["cart"] = []
    return redirect(url_for("home", checkout="done"))


# ----------------------------------------------------------------------------
# Auth (lightweight: pick a customer, no passwords)
# ----------------------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uid = request.form.get("user_id", type=int)
        if uid is None:                                  # "shop as new customer"
            nxt = db.query("SELECT COALESCE(MAX(user_id),0)+1 AS n FROM customers")
            uid = int(nxt["n"].iloc[0])
            db.execute("INSERT INTO customers (user_id, name) VALUES (:u, :n)",
                       {"u": uid, "n": f"Guest {uid}"})
        name = db.query("SELECT name FROM customers WHERE user_id = :u",
                        {"u": uid})
        session["user_id"] = uid
        session["user_name"] = name["name"].iloc[0] if not name.empty \
            else f"Customer {uid}"
        return redirect(url_for("home"))

    sample = db.query(
        "SELECT c.user_id, c.name, COUNT(*) purchases "
        "FROM customers c JOIN interaction_logs l ON l.user_id = c.user_id "
        "GROUP BY c.user_id, c.name ORDER BY purchases DESC LIMIT 8")
    return render_template("login.html", sample=sample.to_dict(orient="records"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# ----------------------------------------------------------------------------
# Admin
# ----------------------------------------------------------------------------
@app.route("/admin")
def admin():
    ready = db.table_exists("kpis")
    return render_template("admin.html", ready=ready)


@app.route("/api/admin/kpis")
def api_admin_kpis():
    return jsonify(db.read_table("kpis").iloc[0].to_dict())


@app.route("/api/admin/forecast-scopes")
def api_forecast_scopes():
    df = db.query("SELECT DISTINCT scope, label FROM forecasts")
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/admin/forecast")
def api_forecast():
    label = request.args.get("label", "All categories")
    df = db.query("SELECT date, kind, value, lower, upper FROM forecasts "
                  "WHERE label = :l ORDER BY date", {"l": label})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    hist = df[df["kind"] == "history"].tail(120)
    fut = df[df["kind"] == "forecast"]
    bridge = hist["value"].iloc[-1] if not hist.empty else None
    return jsonify({
        "labels": hist["date"].tolist() + fut["date"].tolist(),
        "actual": hist["value"].tolist() + [None] * len(fut),
        "forecast": [None] * (len(hist) - 1) + [bridge] + fut["value"].tolist(),
        "lower": [None] * len(hist) + fut["lower"].tolist(),
        "upper": [None] * len(hist) + fut["upper"].tolist(),
        "mape": float(df["mape"].iloc[0]) if "mape" in df and not df.empty else None,
    })


@app.route("/api/admin/inventory")
def api_inventory():
    action = request.args.get("action", "restock")
    order = "reorder_qty DESC" if action == "restock" else \
            ("suggested_discount DESC" if action == "clearance" else "days_of_stock DESC")
    where = "" if action == "all" else "WHERE action = :a"
    df = db.query(f"SELECT name, category, stock_qty, velocity, days_of_stock, "
                  f"est_demand_30d, days_since_sold, reorder_qty, "
                  f"suggested_discount, price, action FROM inventory {where} "
                  f"ORDER BY {order} LIMIT 25",
                  {"a": action} if where else None)
    df["emoji"] = df["category"].map(emoji)
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/admin/category-revenue")
def api_category_revenue():
    df = db.query(
        "SELECT category, SUM(revenue) revenue FROM sales_daily_category "
        "GROUP BY category ORDER BY revenue DESC")
    return jsonify(df.to_dict(orient="records"))


@app.route("/admin/retrain", methods=["POST"])
def admin_retrain():
    """Run the ETL + model pipeline on-demand so recs/forecasts refresh.

    Guarded so a low-memory host (e.g. a free 512MB instance, where the dense
    item-similarity matrix can exceed RAM) returns a friendly message instead
    of crashing the web worker.
    """
    from src import pipeline
    try:
        pipeline.stage_etl_recommendations()
        pipeline.stage_inventory()
        pipeline.stage_build_kpis()
    except MemoryError:
        return jsonify({"ok": False, "error": "Not enough memory to retrain on "
                        "this host. Retrain locally with `python run.py pipeline`."}), 200
    n = db.query("SELECT COUNT(*) n FROM interaction_logs WHERE source='live'")
    return jsonify({"ok": True, "live_events": int(n["n"].iloc[0])})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
