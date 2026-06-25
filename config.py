"""Central configuration for the E-Tail Predictive Intelligence Engine."""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
WAREHOUSE_PATH = DATA_DIR / "warehouse.db"
APP_DB_PATH = DATA_DIR / "app.db"          # the live e-commerce application DB

for _d in (DATA_DIR, RAW_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# SQLAlchemy / SQLite connection strings
WAREHOUSE_URI = f"sqlite:///{WAREHOUSE_PATH.as_posix()}"
APP_DB_URI = f"sqlite:///{APP_DB_PATH.as_posix()}"

# ----------------------------------------------------------------------------
# Data source: the real Kaggle "Online Retail II" dataset (UCI mirror).
# The loader auto-detects a CSV or a zip named like "online_retail*" / "archive"
# inside RAW_DIR or the user's Downloads folder.
# ----------------------------------------------------------------------------
ONLINE_RETAIL_CSV = RAW_DIR / "online_retail_II.csv"
KAGGLE_DATASET = "mashlyn/online-retail-ii-uci"
RANDOM_SEED = 42

# ----------------------------------------------------------------------------
# Modeling parameters
# ----------------------------------------------------------------------------
CF_TOP_N = 10                     # recommendations stored per user
CF_MIN_INTERACTIONS = 3           # ignore ultra-sparse users when scoring
FORECAST_HORIZON = 30             # days to forecast forward

# SARIMA (seasonal ARIMA) — the weekly (7-day) seasonal term lets the forecast
# keep projecting the weekly sales rhythm instead of flat-lining to the mean.
SARIMA_ORDER = (2, 1, 1)              # (p, d, q)         non-seasonal part
SARIMA_SEASONAL_ORDER = (1, 1, 1, 7)  # (P, D, Q, m=7)    weekly seasonality
# Fallback order for short series that can't support the seasonal fit.
ARIMA_FALLBACK_ORDER = (1, 1, 1)

# Anomaly detection
ANOMALY_Z_THRESHOLD = 3.0         # robust z-score cutoff (MAD-based)

# Interaction event weights (implicit feedback -> rating signal)
EVENT_WEIGHTS = {
    "search": 0.5,
    "view": 1.0,
    "add_to_cart": 3.0,
    "purchase": 5.0,
}

# ----------------------------------------------------------------------------
# Catalog: map raw product descriptions -> shopper-friendly categories
# ----------------------------------------------------------------------------
# Checked top-to-bottom; first keyword hit wins. Tuned for the Online Retail II
# gift/homeware catalog. Everything unmatched falls into "General".
CATEGORY_KEYWORDS = [
    ("Seasonal",        ["CHRISTMAS", "XMAS", "ADVENT", "SANTA", "REINDEER",
                          "SNOW", "EASTER", "VALENTINE", "HALLOWEEN"]),
    ("Kitchen & Dining", ["MUG", "CUP", "BOWL", "PLATE", "JUG", "TEAPOT", "TEA ",
                          "CAKE", "BAKING", "JAR", "BOTTLE", "CUTLERY", "NAPKIN",
                          "KITCHEN", "TRAY", "GLASS", "SPOON", "COASTER"]),
    ("Home Decor",      ["HEART", "T-LIGHT", "TLIGHT", "CANDLE", "LANTERN",
                          "FRAME", "CLOCK", "LAMP", "CUSHION", "HANGING",
                          "DECORATION", "MIRROR", "VASE", "ORNAMENT", "DOORMAT"]),
    ("Bags & Travel",   ["BAG", "PURSE", "WALLET", "UMBRELLA", "LUNCH BOX",
                          "SUITCASE", "TOTE"]),
    ("Stationery & Gift", ["CARD", "NOTEBOOK", "PEN ", "PENCIL", "WRAP", "GIFT",
                          "CHALK", "JOURNAL", "STICKER", "ENVELOPE", "TAPE"]),
    ("Garden & Outdoor", ["GARDEN", "PLANT", "FLOWER", "BIRD", "WATERING",
                          "SEED", "OUTDOOR", "PARASOL"]),
    ("Toys & Games",    ["TOY", "GAME", "PLAY", "DOLL", "PUZZLE", "SPACEBOY",
                          "SOLDIER", "SKITTLE", "BINGO"]),
    ("Jewellery",       ["NECKLACE", "BRACELET", "RING", "EARRING", "JEWEL",
                          "BEAD", "BROOCH"]),
    ("Lighting",        ["LIGHT", "LED", "FAIRY", "BULB"]),
]
DEFAULT_CATEGORY = "General"

# ----------------------------------------------------------------------------
# Inventory & "slow mover" / perishability engine
# ----------------------------------------------------------------------------
# Per-category shelf life (days). Past this with no sale, stock is "stale" and
# the engine suggests a clearance discount. Seasonal goods go stale fastest.
CATEGORY_SHELF_LIFE = {
    "Seasonal": 120,
    "Garden & Outdoor": 180,
    "Toys & Games": 365,
    "Stationery & Gift": 365,
    "Kitchen & Dining": 540,
    "Home Decor": 540,
    "Bags & Travel": 540,
    "Lighting": 720,
    "Jewellery": 720,
    "General": 540,
}
DEFAULT_SHELF_LIFE = 540

REORDER_COVER_DAYS = 60       # target days of stock to hold
REORDER_LEAD_DAYS = 30        # restock if projected to run out within this window
VELOCITY_WINDOW_DAYS = 60     # window for measuring recent sales velocity
MAX_CLEARANCE_DISCOUNT = 0.5  # cap auto-suggested clearance discount at 50%
