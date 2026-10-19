"""Data layer for RetailPulse.

Pipeline:  raw UCI / Kaggle "Online Retail" file  ->  clean()  ->  enrich()  ->  processed CSV

The raw file only has 8 columns (InvoiceNo, StockCode, Description, Quantity,
InvoiceDate, UnitPrice, CustomerID, Country).  enrich() adds ~25 own columns
(category, channel, promo, cost, profit, loyalty tier, lead time, ...).
Every engineered column is generated DETERMINISTICALLY (hash based), so the same
input always gives the same output.  They are simulated business attributes and
must be described as such in the project report.

Run from the project root:  python -m src.data --build
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_PATH = ROOT / "data" / "processed" / "retailpulse_enriched.csv.gz"
SRC_PATH = ROOT / "data" / "processed" / "source.txt"

CATEGORY_KEYWORDS = {
    "Home Decor": ["HEART", "FRAME", "CANDLE", "LANTERN", "SIGN", "CLOCK", "MIRROR", "VASE", "DECORATION", "T-LIGHT", "CUSHION"],
    "Kitchen & Dining": ["MUG", "CUP", "PLATE", "BOWL", "TEA", "JAR", "CAKE", "SPOON", "KITCHEN", "LUNCH", "BOTTLE", "TIN"],
    "Gifts & Cards": ["CARD", "WRAP", "GIFT", "RIBBON", "TAG", "BAG", "BOX"],
    "Toys & Games": ["TOY", "GAME", "DOLL", "PUZZLE", "SPACEBOY", "CHILDREN", "PLAY", "DOLLY"],
    "Stationery": ["PENCIL", "NOTEBOOK", "PEN ", "STICKER", "PAPER", "ERASER", "RULER"],
    "Garden & Outdoor": ["GARDEN", "BIRD", "PLANT", "FLOWER", "UMBRELLA", "BUNTING", "PICNIC"],
    "Fashion & Accessories": ["NECKLACE", "BRACELET", "EARRING", "SCARF", "PURSE", "HAIR", "BAG CHARM"],
}
COST_RATIO = {
    "Home Decor": 0.58, "Kitchen & Dining": 0.62, "Gifts & Cards": 0.50, "Toys & Games": 0.60,
    "Stationery": 0.55, "Garden & Outdoor": 0.64, "Fashion & Accessories": 0.52, "General Merchandise": 0.66,
}
EU = {"Austria", "Belgium", "Channel Islands", "Cyprus", "Denmark", "EIRE", "Finland", "France", "Germany",
      "Greece", "Iceland", "Italy", "Lithuania", "Malta", "Netherlands", "Norway", "Poland", "Portugal",
      "Spain", "Sweden", "Switzerland", "European Community", "Czech Republic"}
HOLIDAYS = pd.to_datetime([
    "2009-12-25", "2009-12-28", "2010-01-01", "2010-04-02", "2010-04-05", "2010-05-03", "2010-05-31",
    "2010-08-30", "2010-12-27", "2010-12-28", "2011-01-03", "2011-04-22", "2011-04-25", "2011-04-29",
    "2011-05-02", "2011-05-30", "2011-08-29", "2011-12-26", "2011-12-27"])
NON_PRODUCT_CODES = {"POST", "D", "M", "BANK CHARGES", "AMAZONFEE", "DOT", "CRUK", "S", "B", "PADS", "C2",
                     "TEST001", "TEST002", "AMAZONFEE"}

# (column, origin, description) - column dictionary (documents which columns are original, engineered or simulated)
COLUMN_DICT = [
    ("InvoiceNo", "Original", "Invoice id (prefix C = cancellation / return)"),
    ("StockCode", "Original", "Product id"),
    ("Description", "Original", "Product name (cleaned, upper-case)"),
    ("Quantity", "Original", "Units on the invoice line"),
    ("InvoiceDate", "Original", "Invoice timestamp"),
    ("UnitPrice", "Original", "Selling price per unit (GBP)"),
    ("CustomerID", "Original", "Customer id (missing for guest checkouts)"),
    ("Country", "Original", "Customer country"),
    ("Return_Flag", "Engineered", "True for cancellations / returns"),
    ("Category", "Engineered", "Product category parsed from the description keywords"),
    ("Region", "Engineered", "UK / Europe / Rest of World grouped from Country"),
    ("Channel", "Engineered", "Online / In-Store / Mobile App (simulated, per invoice)"),
    ("Payment_Method", "Engineered", "Credit / Debit / PayPal / Gift Card (simulated)"),
    ("Year / Month / Quarter / DayOfWeek / DayName / Hour / IsWeekend", "Engineered", "Calendar features from InvoiceDate"),
    ("Holiday_Flag", "Engineered", "UK bank holiday indicator"),
    ("Season_Event", "Engineered", "Black Friday & Christmas rush, January sales, Spring, Summer, Regular"),
    ("Promo_Flag", "Engineered", "Promotion running on that day (simulated calendar)"),
    ("Discount_Pct", "Engineered", "Discount percentage on promo days (5 / 10 / 15 / 20)"),
    ("Revenue", "Engineered", "Quantity x UnitPrice (gross)"),
    ("Net_Revenue", "Engineered", "Revenue after discount"),
    ("Supplier", "Engineered", "Supplier A-E per product (simulated)"),
    ("Lead_Time_Days", "Engineered", "Supplier lead time per product, 3-20 days (simulated)"),
    ("UnitCost", "Engineered", "Cost price = UnitPrice x category cost ratio"),
    ("Cost / Profit", "Engineered", "Quantity x UnitCost and Net_Revenue - Cost"),
    ("Satisfaction", "Engineered", "1-5 order rating (simulated, lower for returns)"),
    ("Age_Group / Gender / Acquisition_Channel", "Engineered", "Customer demographics (simulated, per customer)"),
    ("Loyalty_Tier", "Engineered", "Bronze / Silver / Gold / Platinum from the customer's revenue percentile"),
    ("Margin_Pct", "Engineered", "Profit / Net_Revenue x 100 (calculated)"),
    ("Price_Band", "Engineered", "Budget (<2) / Mid (2-5) / Premium (5+) by UnitPrice (calculated)"),
    ("Basket_Value", "Engineered", "Total Net_Revenue of the whole invoice (calculated)"),
    ("Items_Per_Invoice", "Engineered", "Total units on the whole invoice (calculated)"),
    ("Order_Size_Band", "Engineered", "Small (<150) / Medium (<300) / Large (<600) / Bulk by Basket_Value (calculated)"),
    ("Customer_Type", "Engineered", "New (first purchase day) / Returning / Guest (calculated)"),
    ("Days_Since_Prev_Order", "Engineered", "Days since the customer's previous invoice; empty for first order and guests (calculated)"),
    ("Shipping_Cost", "Engineered", "Shipping fee by region; free for in-store, returns, baskets >= 300 (simulated)"),
    ("Delivery_Days", "Engineered", "Days to deliver by region plus random delay; 0 for in-store and returns (simulated)"),
    ("Delivery_Status", "Engineered", "On Time / Late / In-Store Pickup / Returned (simulated)"),
    ("Return_Reason", "Engineered", "Damaged / Wrong Item / Not as Described / Changed Mind / Arrived Late; Not a Return otherwise (simulated)"),
]


# ----------------------------------------------------------------------------- helpers
def _u(values: pd.Series, salt: str) -> np.ndarray:
    """Deterministic pseudo-random number in [0,1) per value."""
    h = pd.util.hash_array(values.astype(str).add(salt).to_numpy(dtype=object))
    return (h % 100000) / 100000.0


def _map_u(series: pd.Series, salt: str) -> pd.Series:
    uniq = pd.Series(series.unique())
    lookup = dict(zip(uniq, _u(uniq, salt)))
    return series.map(lookup).astype(float)


def _categorize(desc: str) -> str:
    for cat, words in CATEGORY_KEYWORDS.items():
        if any(w in desc for w in words):
            return cat
    return "General Merchandise"


# ----------------------------------------------------------------------------- raw loading
def _find_raw() -> Path | None:
    for p in sorted(RAW_DIR.glob("*")):
        if p.suffix.lower() in {".csv", ".xlsx", ".xls"}:
            return p
    return None


def load_raw(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")
    else:
        df = pd.concat(pd.read_excel(path, sheet_name=None).values(), ignore_index=True)
    df = df.rename(columns={"Invoice": "InvoiceNo", "Price": "UnitPrice", "Customer ID": "CustomerID"})
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["Description"] = df["Description"].fillna("UNKNOWN ITEM").astype(str).str.strip().str.upper()
    df = df.drop_duplicates()
    df = df[df["UnitPrice"] > 0]
    df = df[~df["StockCode"].isin(NON_PRODUCT_CODES)]
    df = df[df["Quantity"].abs() <= 5000]  # extreme bulk / reversal outliers
    df["Return_Flag"] = df["InvoiceNo"].str.startswith("C") | (df["Quantity"] < 0)
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------- enrichment
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df["InvoiceDate"]

    # product attributes
    cat_map = {d: _categorize(d) for d in df["Description"].unique()}
    df["Category"] = df["Description"].map(cat_map)
    df["Region"] = np.where(df["Country"] == "United Kingdom", "UK", np.where(df["Country"].isin(EU), "Europe", "Rest of World"))

    # calendar
    df["Year"], df["Month"], df["Quarter"] = ts.dt.year, ts.dt.month, ts.dt.quarter
    df["DayOfWeek"], df["DayName"], df["Hour"] = ts.dt.dayofweek, ts.dt.day_name(), ts.dt.hour
    df["IsWeekend"] = df["DayOfWeek"] >= 5
    day = ts.dt.normalize()
    df["Holiday_Flag"] = day.isin(HOLIDAYS)
    md = ts.dt.month * 100 + ts.dt.day
    df["Season_Event"] = np.select(
        [(md >= 1115) & (md <= 1224), (md >= 102) & (md <= 131), (md >= 301) & (md <= 531), (md >= 601) & (md <= 831)],
        ["Black Friday & Christmas Rush", "January Sales", "Spring", "Summer"], default="Regular")

    # invoice-level attributes
    u = _map_u(df["InvoiceNo"], "channel")
    df["Channel"] = np.select([u < 0.62, u < 0.88], ["Online", "In-Store"], default="Mobile App")
    u = _map_u(df["InvoiceNo"], "pay")
    df["Payment_Method"] = np.select([u < 0.5, u < 0.8, u < 0.93], ["Credit Card", "Debit Card", "PayPal"], default="Gift Card")

    # promotions (calendar-level)
    ud = _map_u(day.dt.strftime("%Y-%m-%d"), "promo")
    df["Promo_Flag"] = ((ud < 0.12) | (df["Season_Event"] == "Black Friday & Christmas Rush")) & ~df["Return_Flag"]
    u = _map_u(df["InvoiceNo"], "disc")
    disc = np.select([u < 0.4, u < 0.75, u < 0.93], [5, 10, 15], default=20)
    df["Discount_Pct"] = np.where(df["Promo_Flag"], disc, 0).astype(float)
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["Net_Revenue"] = df["Revenue"] * (1 - df["Discount_Pct"] / 100)

    # product-level supply attributes
    u = _map_u(df["StockCode"], "supplier")
    df["Supplier"] = np.select([u < .25, u < .5, u < .7, u < .88], ["Supplier A", "Supplier B", "Supplier C", "Supplier D"], default="Supplier E")
    u = _map_u(df["StockCode"], "lead")
    df["Lead_Time_Days"] = (3 + np.floor(u * 18)).astype(int)
    u = _map_u(df["StockCode"], "cost")
    ratio = df["Category"].map(COST_RATIO) + (u - 0.5) * 0.10
    df["UnitCost"] = (df["UnitPrice"] * ratio).round(3)
    df["Cost"] = df["Quantity"] * df["UnitCost"]
    df["Profit"] = df["Net_Revenue"] - df["Cost"]

    # satisfaction (per invoice)
    u = _map_u(df["InvoiceNo"], "csat")
    rating = np.select([u < .04, u < .09, u < .20, u < .55], [1, 2, 3, 4], default=5)
    df["Satisfaction"] = np.where(df["Return_Flag"], np.minimum(rating, 3), rating).astype(float)

    # customer-level attributes
    cid = df["CustomerID"].fillna(-1).astype(int).astype(str)
    guest = df["CustomerID"].isna()
    u = _map_u(cid, "age")
    df["Age_Group"] = np.select([u < .12, u < .38, u < .62, u < .82], ["18-24", "25-34", "35-44", "45-54"], default="55+")
    u = _map_u(cid, "gender")
    df["Gender"] = np.select([u < .52, u < .98], ["Female", "Male"], default="Other")
    u = _map_u(cid, "acq")
    df["Acquisition_Channel"] = np.select([u < .35, u < .55, u < .72, u < .87], ["Organic", "Social Media", "Referral", "Email"], default="Paid Search")
    rev = df.loc[~df["Return_Flag"]].groupby("CustomerID")["Net_Revenue"].sum()
    pct = rev.rank(pct=True)
    tier = pd.cut(pct, [0, .5, .8, .95, 1.0], labels=["Bronze", "Silver", "Gold", "Platinum"], include_lowest=True)
    df["Loyalty_Tier"] = df["CustomerID"].map(tier.astype(object)).astype(object)
    for c in ["Age_Group", "Gender", "Acquisition_Channel", "Loyalty_Tier"]:
        df.loc[guest, c] = "Guest"
        df[c] = df[c].fillna("Guest")

    # ---- extra columns: calculated from the real data --------------------------------
    df["Margin_Pct"] = np.where(df["Net_Revenue"] != 0, df["Profit"] / df["Net_Revenue"] * 100, 0.0).round(2)
    df["Price_Band"] = np.select([df["UnitPrice"] < 2, df["UnitPrice"] < 5], ["Budget", "Mid"], default="Premium")
    inv = df.groupby("InvoiceNo")
    df["Basket_Value"] = inv["Net_Revenue"].transform("sum").round(2)
    df["Items_Per_Invoice"] = inv["Quantity"].transform(lambda q: q.abs().sum()).astype(int)
    bv = df["Basket_Value"].abs()
    df["Order_Size_Band"] = np.select([bv < 150, bv < 300, bv < 600], ["Small", "Medium", "Large"], default="Bulk")

    first_day = df.groupby("CustomerID")["InvoiceDate"].transform("min").dt.normalize()
    df["Customer_Type"] = np.where(guest, "Guest", np.where(day == first_day, "New", "Returning"))
    orders = (df.loc[~guest, ["CustomerID", "InvoiceNo", "InvoiceDate"]]
              .drop_duplicates("InvoiceNo").sort_values(["CustomerID", "InvoiceDate"]))
    orders["Days_Since_Prev_Order"] = orders.groupby("CustomerID")["InvoiceDate"].diff().dt.days
    df["Days_Since_Prev_Order"] = df["InvoiceNo"].map(orders.set_index("InvoiceNo")["Days_Since_Prev_Order"])

    # ---- extra columns: simulated (deterministic, per invoice) -----------------------
    base_days = df["Region"].map({"UK": 2, "Europe": 4, "Rest of World": 8}).astype(int)
    in_store = df["Channel"] == "In-Store"
    u = _map_u(df["InvoiceNo"], "ship")
    ship = df["Region"].map({"UK": 3.5, "Europe": 9.0, "Rest of World": 16.0}) + np.floor(u * 4) * 1.25
    free_ship = df["Basket_Value"] >= 300
    df["Shipping_Cost"] = np.where(in_store | free_ship | df["Return_Flag"], 0.0, ship).round(2)

    u1, u2 = _map_u(df["InvoiceNo"], "deliv"), _map_u(df["InvoiceNo"], "late")
    delay = np.where(u2 < 0.12, 1 + np.floor(u1 * 4), 0)
    days = base_days + np.floor(u1 * 3) + delay
    df["Delivery_Days"] = np.where(in_store | df["Return_Flag"], 0, days).astype(int)
    df["Delivery_Status"] = np.select(
        [df["Return_Flag"], in_store, df["Delivery_Days"] > base_days + 2],
        ["Returned", "In-Store Pickup", "Late"], default="On Time")

    u = _map_u(df["InvoiceNo"], "retreason")
    reason = np.select([u < .30, u < .55, u < .75, u < .90], ["Damaged", "Wrong Item", "Not as Described", "Changed Mind"], default="Arrived Late")
    df["Return_Reason"] = np.where(df["Return_Flag"], reason, "Not a Return")
    return df


# ----------------------------------------------------------------------------- demo data
def make_demo(seed: int = 42, n_customers: int = 1200, n_products: int = 320) -> pd.DataFrame:
    """Synthetic data in the SAME schema as UCI Online Retail (used only if no raw file is present)."""
    rng = np.random.default_rng(seed)
    adjectives = ["VINTAGE", "RETRO", "PINK", "BLUE", "RED", "GREEN", "IVORY", "WHITE", "SPOTTY", "STRIPED", "ANTIQUE", "JUMBO"]
    nouns = [w.strip() for ws in CATEGORY_KEYWORDS.values() for w in ws[:6]] + ["LAMP", "TRAY", "APRON", "TOWEL"]
    descs = [f"{rng.choice(adjectives)} {rng.choice(nouns)} {rng.choice(['SET', 'DESIGN', 'STYLE', 'LARGE', 'SMALL'])}" for _ in range(n_products)]
    codes = [str(21000 + i) + rng.choice(["", "A", "B"]) for i in range(n_products)]
    prices = np.round(np.exp(rng.normal(0.9, 0.7, n_products)).clip(0.29, 40), 2)
    pop = rng.dirichlet(np.ones(n_products) * 0.35)
    cust_ids = 12346 + rng.choice(5000, n_customers, replace=False)
    activity = rng.gamma(1.2, 1.0, n_customers)
    churn_day = np.where(rng.random(n_customers) < 0.35, rng.integers(60, 340, n_customers), 10_000)
    join_day = np.where(rng.random(n_customers) < 0.4, rng.integers(0, 250, n_customers), 0)
    countries = rng.choice(["United Kingdom", "Germany", "France", "EIRE", "Spain", "Netherlands", "Australia", "Norway", "Japan"],
                           n_customers, p=[.86, .03, .03, .02, .01, .01, .01, .01, .02])
    days = pd.date_range("2010-12-01", "2011-12-09", freq="D")
    dow_w = {0: 1.0, 1: 1.1, 2: 1.1, 3: 1.3, 4: 0.95, 5: 0.0, 6: 0.55}
    rows, inv = [], 536365
    for i, d in enumerate(days):
        season = 1 + 0.8 * (d.month in (10, 11)) + 0.3 * (d.month == 12 and d.day < 10) - 0.2 * (d.month in (1, 2))
        lam = 38 * dow_w[d.dayofweek] * season * (1 + 0.45 * i / len(days))
        for _ in range(rng.poisson(lam)):
            inv += 1
            guest = rng.random() < 0.2
            alive = (churn_day > i) & (join_day <= i)
            if not guest and alive.any():
                w = activity * alive
                ci = rng.choice(n_customers, p=w / w.sum())
                cid, ctry = cust_ids[ci], countries[ci]
            else:
                cid, ctry = np.nan, "United Kingdom"
            ts = d + pd.Timedelta(hours=int(rng.integers(8, 18)), minutes=int(rng.integers(0, 60)))
            is_ret = rng.random() < 0.02
            for p in rng.choice(n_products, int(np.clip(rng.geometric(0.18), 1, 25)), p=pop):
                q = int(np.clip(rng.geometric(0.2), 1, 144))
                rows.append((("C" if is_ret else "") + str(inv), codes[p], descs[p], -q if is_ret else q, ts,
                             prices[p], cid, ctry))
    return pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"])


# ----------------------------------------------------------------------------- public API
def build_dataset(force: bool = False) -> tuple[pd.DataFrame, str]:
    raw_path = _find_raw()
    src = SRC_PATH.read_text() if SRC_PATH.exists() else ""
    if PROC_PATH.exists() and not force and not (raw_path and src.startswith("Demo")):
        return pd.read_csv(PROC_PATH, parse_dates=["InvoiceDate"]), src
    if raw_path:
        raw, src = load_raw(raw_path), f"Real dataset: {raw_path.name} + engineered columns"
    else:
        raw, src = make_demo(), "Demo data: synthetic, same schema as UCI Online Retail (add your file to data/raw/)"
    df = enrich(clean(raw))
    PROC_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROC_PATH, index=False)
    SRC_PATH.write_text(src)
    return df, src


def quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    sales = df[~df["Return_Flag"]]
    checks = [
        ("InvoiceNo has no nulls", df["InvoiceNo"].notna().all()),
        ("UnitPrice > 0 on every row", bool((df["UnitPrice"] > 0).all())),
        ("Quantity is never zero", bool((df["Quantity"] != 0).all())),
        ("Sales rows have Quantity > 0", bool((sales["Quantity"] > 0).all())),
        ("No duplicate rows", not df.duplicated().any()),
        ("InvoiceDate within a valid range", bool(df["InvoiceDate"].between("2009-01-01", "2026-12-31").all())),
        ("Net revenue is never above gross revenue (sales)", bool((sales["Net_Revenue"] <= sales["Revenue"] + 1e-9).all())),
        ("Customer id missing < 30%", df["CustomerID"].isna().mean() < 0.30),
        ("Lead time between 3 and 21 days", bool(df["Lead_Time_Days"].between(3, 21).all())),
        ("Delivery days never negative", bool((df["Delivery_Days"] >= 0).all())),
        ("Shipping cost never negative", bool((df["Shipping_Cost"] >= 0).all())),
        ("Return reason is set only on returns", bool(((df["Return_Reason"] != "Not a Return") == df["Return_Flag"]).all())),
    ]
    return pd.DataFrame({"Check": [c for c, _ in checks], "Status": ["Pass" if ok else "Fail" for _, ok in checks]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="rebuild data/processed from data/raw")
    a = ap.parse_args()
    d, s = build_dataset(force=a.build)
    print(s, d.shape)
