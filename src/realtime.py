"""Real-time analytics engine: a live order stream, running aggregates, alerts and live stock levels.

The public dataset has no live shop feed, so the stream REPLAYS real historical orders (real products, prices,
countries and basket sizes) at a rate you choose. Everything downstream (ingest, running totals, alerting,
stock depletion, dashboard) works the same way it would on a real feed such as Kafka or a webhook.

Pure Python (no Streamlit) so it can be unit-tested; the app keeps one shared instance (see src/services.py).
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from datetime import datetime

import numpy as np
import pandas as pd

LINE_COLS = ["InvoiceNo", "StockCode", "Description", "Category", "Quantity", "Net_Revenue", "Profit",
             "Country", "Region", "Channel", "CustomerID"]
MAX_CATCHUP = 30.0          # seconds of orders generated after the page was closed (prevents a burst on return)
BUCKET = 30                 # seconds per bucket in the traffic-spike test
ANOMALY_COOLDOWN = 120.0    # seconds between two spike alerts


class LiveStream:
    def __init__(self, df: pd.DataFrame, policy: pd.DataFrame, seed: int | None = None):
        sales = df.loc[~df["Return_Flag"], LINE_COLS].sort_values("InvoiceNo", kind="stable")
        self._c = {c: sales[c].to_numpy() for c in LINE_COLS}
        inv = self._c["InvoiceNo"]
        self._starts = np.flatnonzero(np.r_[True, inv[1:] != inv[:-1]])
        self._ends = np.r_[self._starts[1:], len(inv)]
        baskets = np.add.reduceat(self._c["Net_Revenue"].astype(float), self._starts)
        self.big_order = float(np.quantile(baskets, 0.95))           # "high-value order" = top 5% of baskets
        self._pol = {r.StockCode: (r.Description, float(r.Reorder_Point), float(r.EOQ), float(r.AvgDailyDemand))
                     for r in policy.itertuples(index=False)}
        self._stock0 = {r.StockCode: float(r.Stock_On_Hand) for r in policy.itertuples(index=False)}
        self._lock = threading.RLock()
        self._rng = np.random.default_rng(seed)
        self.rate = 6                                                 # orders per minute (about one trading day per 11 minutes)
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        with self._lock:
            self.running = False
            self.orders: deque = deque(maxlen=20_000)
            self.alerts: deque = deque(maxlen=300)
            self.totals = dict(orders=0, units=0, revenue=0.0, profit=0.0)
            self._by_region, self._by_cat = defaultdict(float), defaultdict(float)
            self._prod: dict = {}                                     # code -> [description, units, revenue]
            self._stock = dict(self._stock0)
            self._low = {c for c, s in self._stock.items() if s <= self._pol[c][1]}
            self._out = {c for c, s in self._stock.items() if s <= 0}
            self._seq, self._first_ts = 0, float("inf")
            self._last_tick = None
            self._last_anom = 0.0
            self._surge_until, self._surge_factor = 0.0, 1.0

    def set_rate(self, rate: int) -> None:
        self.rate = int(rate)

    def start(self, now: float | None = None) -> None:
        with self._lock:
            now = now or time.time()
            if not self.running:
                self.running, self._last_tick = True, now
            if not self.orders:
                self.warm_up(10, now)

    def pause(self) -> None:
        with self._lock:
            self.running = False

    def warm_up(self, minutes: float, now: float | None = None) -> None:
        """Back-fill recent history so charts and the spike baseline are not empty at start."""
        with self._lock:
            now = now or time.time()
            n = int(self._rng.poisson(self.rate * minutes))
            self.ingest(sorted(now - self._rng.random(n) * minutes * 60))

    def surge(self, factor: float = 4.0, seconds: float = 90.0, now: float | None = None) -> None:
        with self._lock:
            now = now or time.time()
            self._surge_until, self._surge_factor = now + seconds, factor
            self._alert(now, "info", "Simulation", f"Demand surge started: orders x{factor:g} for {seconds:g} seconds")

    # ------------------------------------------------------------------ ingest
    def advance(self, now: float | None = None) -> int:
        """Generate the orders that 'arrived' since the last call. Returns how many."""
        with self._lock:
            if not self.running:
                return 0
            now = now or time.time()
            elapsed = min(max(now - self._last_tick, 0.0), MAX_CATCHUP)
            self._last_tick = now
            rate = self.rate * (self._surge_factor if now < self._surge_until else 1.0)
            n = int(self._rng.poisson(rate / 60 * elapsed))
            self.ingest(sorted(now - self._rng.random(n) * elapsed))
            self.detect_anomaly(now)
            return n

    def ingest(self, timestamps) -> None:
        with self._lock:
            for ts in timestamps:
                self._one_order(float(ts))

    def _one_order(self, ts: float) -> None:
        k = int(self._rng.integers(len(self._starts)))
        a, b, c = self._starts[k], self._ends[k], self._c
        codes, desc, cat = c["StockCode"][a:b], c["Description"][a:b], c["Category"][a:b]
        qty, rev, prof = c["Quantity"][a:b], c["Net_Revenue"][a:b], c["Profit"][a:b]
        self._seq += 1
        cust = c["CustomerID"][a]
        order = dict(ts=ts, order_id=f"L{self._seq:06d}", customer="Guest" if pd.isna(cust) else str(int(cust)),
                     country=c["Country"][a], region=c["Region"][a], channel=c["Channel"][a], items=int(b - a),
                     units=int(qty.sum()), revenue=float(rev.sum()), profit=float(prof.sum()))
        self.orders.append(order)
        self._first_ts = min(self._first_ts, ts)
        t = self.totals
        t["orders"] += 1
        t["units"] += order["units"]
        t["revenue"] += order["revenue"]
        t["profit"] += order["profit"]
        self._by_region[order["region"]] += order["revenue"]
        for code, d, ct, q, r in zip(codes, desc, cat, qty, rev):
            p = self._prod.setdefault(code, [d, 0, 0.0])
            p[1] += int(q)
            p[2] += float(r)
            self._by_cat[ct] += float(r)
            self._take_stock(code, float(q), ts)
        if order["revenue"] >= self.big_order:
            self._alert(ts, "info", "High-value order",
                        f"{order['order_id']} from {order['country']}: £{order['revenue']:,.0f} ({order['items']} items)")

    def _take_stock(self, code: str, qty: float, ts: float) -> None:
        s = self._stock.get(code)
        if s is None:
            return
        new = max(0.0, s - qty)
        self._stock[code] = new
        desc, rop, _, _ = self._pol[code]
        if new <= 0 and code not in self._out:
            self._out.add(code)
            self._low.add(code)
            self._alert(ts, "critical", "Stockout", f"{desc} is out of stock")
        elif new <= rop and code not in self._low:
            self._low.add(code)
            self._alert(ts, "warning", "Reorder point", f"{desc}: {int(new)} left (reorder point {int(rop)})")

    def _alert(self, ts: float, severity: str, kind: str, message: str) -> None:
        self.alerts.append(dict(ts=ts, severity=severity, type=kind, message=message))

    # ------------------------------------------------------------------ anomaly detection
    def detect_anomaly(self, now: float | None = None) -> bool:
        """Traffic-spike test: is the last minute's order count far above the Poisson expectation from the 5 minutes before?"""
        with self._lock:
            now = now or time.time()
            if now - self._last_anom < ANOMALY_COOLDOWN or not self.orders:
                return False
            recent_start = now - 2 * BUCKET
            base_start = recent_start - 10 * BUCKET
            ts = []
            for o in reversed(self.orders):
                if o["ts"] < base_start:
                    break
                ts.append(o["ts"])
            ts = np.array(ts)
            span = recent_start - max(base_start, self._first_ts)
            if span < 6 * BUCKET:
                return False
            n_recent = int((ts >= recent_start).sum())
            expected = float(((ts >= base_start) & (ts < recent_start)).sum()) / span * 2 * BUCKET
            if expected < 4:
                return False
            z = (n_recent - expected) / np.sqrt(expected)
            if z >= 4 and n_recent >= 1.5 * expected:
                self._last_anom = now
                per_min = 60 / (2 * BUCKET)
                self._alert(now, "critical", "Demand spike",
                            f"{n_recent * per_min:.0f} orders/min vs about {expected * per_min:.0f} normally (z = {z:.1f})")
                return True
            return False

    # ------------------------------------------------------------------ actions
    def restock(self) -> int:
        """Place purchase orders for every product at or below its reorder point (order up to ROP + EOQ, instant delivery in this demo)."""
        with self._lock:
            n = 0
            for code in list(self._low):
                _, rop, eoq, _ = self._pol[code]
                self._stock[code] = rop + eoq
                self._low.discard(code)
                self._out.discard(code)
                n += 1
            if n:
                self._alert(time.time(), "info", "Purchase orders", f"Stock ordered for {n:,} products")
            return n

    # ------------------------------------------------------------------ read side (what the dashboard shows)
    def kpis(self, now: float | None = None) -> dict:
        with self._lock:
            now = now or time.time()
            r5 = p5 = 0.0
            n2 = 0
            for o in reversed(self.orders):
                if o["ts"] < now - 600:
                    break
                if o["ts"] >= now - 300:
                    r5 += o["revenue"]
                else:
                    p5 += o["revenue"]
                n2 += o["ts"] >= now - 120
            t = self.totals
            return dict(**t, aov=t["revenue"] / t["orders"] if t["orders"] else 0.0, rev5=r5, prev5=p5, per_min=n2 / 2)

    def series(self, bucket_s: int = 60, window_min: int = 30, now: float | None = None) -> pd.DataFrame:
        with self._lock:
            now = now or time.time()
            first = int((now - window_min * 60) // bucket_s) * bucket_s
            idx = np.arange(first, int(now // bucket_s) * bucket_s + bucket_s, bucket_s)
            rev, cnt = np.zeros(len(idx)), np.zeros(len(idx))
            for o in reversed(self.orders):
                if o["ts"] < first:
                    break
                j = int((o["ts"] - first) // bucket_s)
                if 0 <= j < len(idx):
                    rev[j] += o["revenue"]
                    cnt[j] += 1
        return pd.DataFrame({"Time": [datetime.fromtimestamp(float(t)) for t in idx], "Revenue": rev, "Orders": cnt})

    def recent_orders(self, n: int = 20) -> pd.DataFrame:
        with self._lock:
            rows = list(self.orders)[-n:][::-1]
        cols = ["Time", "Order", "Customer", "Country", "Channel", "Items", "Units", "Revenue"]
        if not rows:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame({"Time": [datetime.fromtimestamp(o["ts"]).strftime("%H:%M:%S") for o in rows],
                             "Order": [o["order_id"] for o in rows], "Customer": [o["customer"] for o in rows],
                             "Country": [o["country"] for o in rows], "Channel": [o["channel"] for o in rows],
                             "Items": [o["items"] for o in rows], "Units": [o["units"] for o in rows],
                             "Revenue": [o["revenue"] for o in rows]})

    def recent_alerts(self, n: int = 10, kinds: set[str] | None = None) -> list[dict]:
        with self._lock:
            rows = [a for a in reversed(self.alerts) if kinds is None or a["type"] in kinds][:n]
        return [dict(r, time=datetime.fromtimestamp(r["ts"]).strftime("%H:%M:%S")) for r in rows]

    def top_products(self, n: int = 8) -> pd.DataFrame:
        with self._lock:
            rows = sorted(self._prod.values(), key=lambda p: p[1], reverse=True)[:n]
        return pd.DataFrame(rows, columns=["Product", "Units", "Revenue"])

    def by_category(self, n: int = 8) -> pd.DataFrame:
        with self._lock:
            d = dict(self._by_cat)
        return pd.DataFrame({"Category": list(d), "Revenue": list(d.values())}).nlargest(n, "Revenue")

    def by_region(self) -> pd.DataFrame:
        with self._lock:
            d = dict(self._by_region)
        return pd.DataFrame({"Region": list(d), "Revenue": list(d.values())})

    def stock_summary(self) -> dict:
        with self._lock:
            return dict(tracked=len(self._stock), low=len(self._low), out=len(self._out))

    def watchlist(self, n: int = 15) -> pd.DataFrame:
        """Products that have sold since the stream started, closest to running out first."""
        with self._lock:
            rows = []
            for code in self._prod:
                if code not in self._stock:      # sold, but not in the inventory table (no recent demand)
                    continue
                desc, rop, _, dem = self._pol[code]
                s = self._stock[code]
                rows.append(dict(Product=desc, Stock=int(s), Reorder_Point=int(rop), Days_Of_Cover=round(s / dem, 1) if dem > 0 else float("inf"),
                                 Pct_Of_Reorder_Point=round(s / rop * 100) if rop > 0 else 999,
                                 Status="Stockout" if s <= 0 else "Reorder now" if s <= rop else "OK"))
        cols = ["Product", "Stock", "Reorder_Point", "Days_Of_Cover", "Pct_Of_Reorder_Point", "Status"]
        return pd.DataFrame(rows, columns=cols).sort_values("Days_Of_Cover").head(n).reset_index(drop=True)
