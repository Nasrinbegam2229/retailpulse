import pytest

from src.data import clean, enrich, make_demo
from src.inventory import compute_policy, product_table
from src.realtime import LiveStream


@pytest.fixture(scope="module")
def parts():
    df = enrich(clean(make_demo(n_customers=300, n_products=120)))
    return df, compute_policy(product_table(df))


@pytest.fixture()
def stream(parts):
    s = LiveStream(*parts, seed=1)
    s.set_rate(12)
    return s


def test_ingest_updates_totals_and_feed(stream):
    stream.ingest([1000.0 + i for i in range(50)])
    k = stream.kpis(now=1060.0)
    assert k["orders"] == 50 and len(stream.recent_orders(100)) == 50
    assert k["revenue"] == pytest.approx(sum(o["revenue"] for o in stream.orders))
    assert stream.top_products()["Units"].is_monotonic_decreasing


def test_stock_drains_never_goes_negative_and_stockout_alerts(stream):
    code = next(iter(stream._stock))
    stream._stock[code] = 1.0
    stream._low.discard(code), stream._out.discard(code)
    stream.ingest([2000.0 + i for i in range(300)])
    assert min(stream._stock.values()) >= 0
    kinds = {a["type"] for a in stream.alerts}
    assert kinds & {"Stockout", "Reorder point"}
    assert stream.stock_summary()["low"] >= 1


def test_restock_clears_flags_and_raises_stock(stream):
    stream.ingest([3000.0 + i for i in range(300)])
    low_before = stream.stock_summary()["low"]
    assert low_before > 0
    assert stream.restock() == low_before
    assert stream.stock_summary() == dict(tracked=len(stream._stock), low=0, out=0)
    assert all(stream._stock[c] > stream._pol[c][1] for c in stream._stock)


def test_advance_only_generates_when_running(stream):
    assert stream.advance(now=100.0) == 0
    stream.set_rate(120)
    stream.start(now=100.0)
    n0 = len(stream.orders)
    assert stream.advance(now=130.0) > 10  # ~60 expected in 30 s
    assert len(stream.orders) > n0
    stream.pause()
    assert stream.advance(now=160.0) == 0


def test_catchup_is_capped(stream):
    stream.set_rate(60)
    stream.start(now=100.0)
    n = stream.advance(now=100.0 + 3600)  # page was closed for an hour
    assert n < 100  # capped at 30 s of orders, not 3,600 s


def test_demand_spike_is_detected_but_normal_traffic_is_not(stream):
    now = 10_000.0
    stream.set_rate(12)
    stream.warm_up(10, now=now)
    assert not stream.detect_anomaly(now)                      # steady traffic: no alert
    stream.ingest([now - 55 + i * 0.4 for i in range(120)])    # 120 orders in the last minute (about 10x normal)
    assert stream.detect_anomaly(now)
    assert any(a["type"] == "Demand spike" for a in stream.alerts)
    assert not stream.detect_anomaly(now + 10)                 # cooldown


def test_reset_clears_everything(stream):
    stream.start()
    stream.reset()
    assert stream.totals["orders"] == 0 and not stream.orders and not stream.alerts and not stream.running


def test_series_has_zero_filled_buckets(stream):
    stream.ingest([5000.0, 5001.0, 5200.0])
    s = stream.series(bucket_s=60, window_min=10, now=5230.0)
    assert s["Orders"].sum() == 3 and (s["Orders"] == 0).any()


def test_products_missing_from_the_inventory_table_do_not_break_the_watchlist(parts):
    df, policy = parts
    s = LiveStream(df, policy.iloc[: len(policy) // 2], seed=3)   # half the products are untracked
    s.ingest([100.0 + i for i in range(200)])
    w = s.watchlist(15)
    assert len(w) > 0 and set(w["Status"]) <= {"OK", "Reorder now", "Stockout"}
    assert s.top_products()["Units"].sum() > 0                    # untracked products still count as sales
