"""Live market intelligence for the rebalancer, aggregated from 5 free
sources into one compact JSON blob for Claude.

Reliability, tested against live endpoints while building this:
  - yfinance: reliable. Quotes, 1y history (for 3mo/6mo momentum),
    expense ratio + top-10 holdings via Ticker.funds_data.
  - pandas-datareader (FRED "SP500" series): reliable, no key needed.
    Used as the S&P 500 benchmark for relative-strength comparisons.
  - stockdex (justETF): justETF is a European UCITS fund database — it
    does not cover most US-domiciled ETFs (VOO, QQQM, etc. all raised
    WrongSecurityType / parse errors in testing). Called best-effort;
    expect it to contribute nothing for a typical US ETF whitelist.
  - finvizfinance screener: returned non-tickers in testing (e.g.
    "OOKTG") and failed on a plain AAPL lookup — looks blocked or broken
    in this environment. Called best-effort; treat any tickers it
    returns as untrusted input.
  - requests + BeautifulSoup (Morningstar): morningstar.com/etfs is
    server-rendered so it's fetchable, but the free page is an editorial
    hub, not a structured ratings table — ETF.com returns a Cloudflare
    challenge page instead of content. This source yields a few
    headline strings as qualitative color, not a ranked ETF list.

Because 3 of the 5 sources are unreliable or return untrusted data, every
externally-surfaced ticker (from finviz or Morningstar) is re-validated
against real yfinance data before it can appear as a buyable candidate —
garbage tickers are silently dropped, never handed to Claude.
"""
import re
from datetime import datetime, timedelta

import requests
import yfinance as yf
from bs4 import BeautifulSoup

try:
    import pandas_datareader.data as pdr_web
except ImportError:
    pdr_web = None

try:
    from stockdex import Ticker as StockdexTicker
except ImportError:
    StockdexTicker = None

try:
    from finvizfinance.screener.performance import Performance as FinvizPerformance
except ImportError:
    FinvizPerformance = None

EXPENSE_RATIO_MAX_PCT = 0.50  # hard cutoff; candidates above this are dropped
OVERLAP_MAX_PCT = 60.0  # hard cutoff; candidates above this are dropped

# Curated fallback candidate universe, banded loosely by risk so Mode 4 has
# something real to work with even when finviz/Morningstar contribute
# nothing (which, per the reliability notes above, is the common case).
# Edit freely — this is a plain list, not a scraped or LLM-sourced one.
CANDIDATE_UNIVERSE = {
    "conservative": ["BND", "AGG", "SHY", "VIG", "SCHD"],
    "moderate": ["VTI", "SCHB", "SPY", "VUG", "VYM"],
    "aggressive": ["QQQ", "XLK", "SOXX", "ARKK", "SMH"],
    "very_aggressive": ["TQQQ", "SOXL", "UPRO"],  # leveraged — only at high risk tolerance
}


def _debug(label: str, data) -> None:
    import os
    import json as _json
    if os.environ.get("DEBUG_API", "true").strip().lower() != "true":
        return
    print(f"\n----- DEBUG: {label} -----")
    print(_json.dumps(data, indent=2, default=str)[:4000])
    print(f"----- end {label} -----\n")


# -- technical indicators (pure math on data already fetched — no extra
#    network calls, computed unconditionally inside yf_snapshot) ------------

def _compute_rsi(closes, period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    delta = closes.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def _sma_deviation(closes, window: int) -> tuple[float | None, str | None]:
    """Returns (% deviation of last close from the SMA, Bollinger-style
    band signal at 2 standard deviations)."""
    if len(closes) < window:
        return None, None
    sma = closes.rolling(window).mean().iloc[-1]
    std = closes.rolling(window).std().iloc[-1]
    last = closes.iloc[-1]
    if not sma:
        return None, None
    deviation_pct = round((last - sma) / sma * 100, 2)
    if std:
        if last > sma + 2 * std:
            band_signal = "above_upper_band"
        elif last < sma - 2 * std:
            band_signal = "below_lower_band"
        else:
            band_signal = "within_bands"
    else:
        band_signal = None
    return deviation_pct, band_signal


def _rsi_signal(rsi: float | None) -> str | None:
    """Thresholds per spec: RSI > 70 overbought, RSI < 38 oversold (not
    the textbook 30 — this app uses the wider band explicitly requested)."""
    if rsi is None:
        return None
    if rsi > 70:
        return "overbought"
    if rsi < 38:
        return "oversold"
    return "neutral"


# -- source 1: yfinance ------------------------------------------------------

def yf_snapshot(ticker: str) -> dict | None:
    """Reliable source. Returns price, expense ratio, category, top-10
    holdings (symbols only), and 3mo/6mo momentum, or None if the ticker
    doesn't resolve to real data — this is also the validation gate for
    tickers surfaced by the less reliable sources below."""
    try:
        t = yf.Ticker(ticker)
        # "1y" (not "6mo") so there's reliably >126 trading days of history
        # to look back from for the 6-month momentum figure below.
        hist = t.history(period="1y", interval="1d")
        if hist.empty or "Close" not in hist:
            return None
        closes = hist["Close"].dropna()
        if len(closes) < 20:
            return None
        last = float(closes.iloc[-1])

        def pct_change_over(trading_days: int) -> float | None:
            if len(closes) <= trading_days:
                return None
            start = float(closes.iloc[-trading_days - 1])
            return round((last - start) / start * 100, 2) if start else None

        expense_ratio = None
        category = None
        top_holdings = []
        top_holdings_weighted = []
        try:
            fd = t.funds_data
            ops = fd.fund_operations
            if ops is not None and "Annual Report Expense Ratio" in ops.index and ticker in ops.columns:
                expense_ratio = round(float(ops.loc["Annual Report Expense Ratio", ticker]) * 100, 3)
            overview = fd.fund_overview or {}
            category = overview.get("categoryName")
            holdings_df = fd.top_holdings
            if holdings_df is not None and not holdings_df.empty:
                top_holdings = list(holdings_df.index[:10])
                top_holdings_weighted = [
                    {"symbol": sym, "weight": float(holdings_df.loc[sym, "Holding Percent"])}
                    for sym in holdings_df.index[:10]
                ]
        except Exception:
            pass  # not every ticker (e.g. individual stocks) has funds_data

        info = {}
        try:
            info = t.info or {}
        except Exception:
            pass

        rsi_14 = _compute_rsi(closes, period=14)
        sma50_dev, band_signal = _sma_deviation(closes, window=50)
        sma200_dev, _ = _sma_deviation(closes, window=200)

        return {
            "ticker": ticker,
            "name": info.get("shortName") or info.get("longName") or ticker,
            "last_price": round(last, 4),
            "category": category,
            "expense_ratio_pct": expense_ratio,
            "trailing_pe": info.get("trailingPE"),
            "momentum_3m_pct": pct_change_over(63),
            "momentum_6m_pct": pct_change_over(126),
            "top_holdings": top_holdings,
            "top_holdings_weighted": top_holdings_weighted,
            "rsi_14": rsi_14,
            "rsi_signal": _rsi_signal(rsi_14),
            "sma50_deviation_pct": sma50_dev,
            "sma200_deviation_pct": sma200_dev,
            "bollinger_signal": band_signal,
        }
    except Exception:
        return None


def yf_forward_lookthrough(top_holdings_weighted: list[dict], cache: dict, top_n: int = 3) -> dict:
    """Mode 5 needs forward P/E / earnings growth / consensus 12mo return
    per ETF, but yfinance's ETF-level .info has none of these — confirmed
    empirically (forwardPE, earningsGrowth, targetMeanPrice are all None
    for VOO/QQQM/SMH). Individual stocks DO carry them. So this looks
    through to an ETF's top-N underlying holdings (by weight) and computes
    a weight-renormalized average — a standard look-through technique, not
    a fabricated number. `cache` is a per-request dict the caller reuses
    across every ETF being evaluated, since mega-caps like AAPL/MSFT/NVDA
    show up as top holdings of many different ETFs — without it this
    would refetch the same stock's .info dozens of times per analyze
    click."""
    if not top_holdings_weighted:
        return {"forward_pe": None, "earnings_growth_pct": None, "implied_12m_return_pct": None, "lookthrough_holdings": []}

    ranked = sorted(top_holdings_weighted, key=lambda h: h["weight"], reverse=True)[:top_n]
    fpe_num = fpe_w = eg_num = eg_w = ret_num = ret_w = 0.0
    for h in ranked:
        sym = h["symbol"]
        if sym not in cache:
            try:
                cache[sym] = yf.Ticker(sym).info or {}
            except Exception:
                cache[sym] = {}
        info = cache[sym]
        w = h["weight"]
        fpe = info.get("forwardPE")
        eg = info.get("earningsGrowth")
        target = info.get("targetMeanPrice")
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if fpe:
            fpe_num += fpe * w
            fpe_w += w
        if eg is not None:
            eg_num += eg * w
            eg_w += w
        if target and price:
            ret_num += (target / price - 1) * 100 * w
            ret_w += w

    return {
        "forward_pe": round(fpe_num / fpe_w, 2) if fpe_w else None,
        "earnings_growth_pct": round(eg_num / eg_w * 100, 2) if eg_w else None,
        "implied_12m_return_pct": round(ret_num / ret_w, 2) if ret_w else None,
        "lookthrough_holdings": [h["symbol"] for h in ranked],
    }


# -- source 2: stockdex (justETF) --------------------------------------------

def stockdex_expense_ratio(ticker: str) -> float | None:
    """Best-effort. justETF barely covers US-domiciled ETFs — expect None
    for most tickers in a typical US whitelist; failures are swallowed."""
    if StockdexTicker is None:
        return None
    try:
        t = StockdexTicker(ticker=ticker, security_type="etf")
        basics = t.justetf_basics
        if isinstance(basics, dict):
            for key in ("ter", "totalExpenseRatio", "expense_ratio"):
                if basics.get(key) is not None:
                    return float(basics[key])
    except Exception:
        pass
    return None


# -- source 3: finvizfinance screener ----------------------------------------

def finviz_momentum_leaders(limit: int = 5) -> list[str]:
    """Best-effort. Observed to return unreliable/garbage tickers in
    testing — callers MUST NOT trust these directly; they're re-validated
    against yfinance before use anywhere downstream."""
    if FinvizPerformance is None:
        return []
    try:
        fv = FinvizPerformance()
        fv.set_filter(filters_dict={"Net Expense Ratio": "Under 1.0%"})
        df = fv.screener_view(order="Performance (Quarter)", limit=limit, ascend=False)
        if df is None or "Ticker" not in df:
            return []
        return [str(t).upper() for t in df["Ticker"].head(limit).tolist()]
    except Exception:
        return []


# -- source 4: pandas-datareader (FRED) --------------------------------------

def sp500_benchmark_momentum() -> dict:
    """Reliable. S&P 500 index level momentum over the same 3mo/6mo
    windows used for candidate momentum, for relative-strength comparison."""
    if pdr_web is None:
        return {"momentum_3m_pct": None, "momentum_6m_pct": None}
    try:
        end = datetime.now()
        start = end - timedelta(days=220)
        df = pdr_web.DataReader("SP500", "fred", start, end).dropna()
        if df.empty:
            return {"momentum_3m_pct": None, "momentum_6m_pct": None}
        series = df["SP500"]
        last = float(series.iloc[-1])

        def pct_change_over(days_back: int) -> float | None:
            cutoff = series.index[-1] - timedelta(days=days_back)
            window = series[series.index <= cutoff]
            if window.empty:
                return None
            start_val = float(window.iloc[-1])
            return round((last - start_val) / start_val * 100, 2) if start_val else None

        return {
            "momentum_3m_pct": pct_change_over(91),
            "momentum_6m_pct": pct_change_over(182),
        }
    except Exception:
        return {"momentum_3m_pct": None, "momentum_6m_pct": None}


def _fred_series_trend(series_id: str) -> tuple[float | None, float | None, str | None]:
    """Returns (latest_value, value_6mo_ago, trend) for a FRED series, where
    trend is "rising"/"falling"/"flat" using a small deadband so daily
    noise doesn't flip the label."""
    if pdr_web is None:
        return None, None, None
    try:
        end = datetime.now()
        start = end - timedelta(days=400)
        df = pdr_web.DataReader(series_id, "fred", start, end).dropna()
        if df.empty:
            return None, None, None
        series = df[series_id]
        latest = float(series.iloc[-1])
        prior_idx = max(0, len(series) - 126)
        prior = float(series.iloc[prior_idx])
        delta = latest - prior
        trend = "rising" if delta > 0.15 else "falling" if delta < -0.15 else "flat"
        return round(latest, 3), round(prior, 3), trend
    except Exception:
        return None, None, None


def macro_regime_signals() -> dict:
    """Mode 6 inputs: 10-year Treasury yield trend and breakeven inflation
    trend (both real FRED series), S&P 500's deviation from its 200-day
    SMA (via SPY as a liquid proxy), and 3-month momentum across four
    sector/factor benchmarks (Technology, Financials, Industrials,
    Small Caps) as a simple relative sector-rotation signal."""
    treasury_now, treasury_6m_ago, treasury_trend = _fred_series_trend("DGS10")
    inflation_now, inflation_6m_ago, inflation_trend = _fred_series_trend("T10YIE")

    spy = yf_snapshot("SPY")
    sp500_vs_200sma_pct = spy.get("sma200_deviation_pct") if spy else None

    sector_momentum = {}
    for symbol, label in (("XLK", "Technology"), ("XLF", "Financials"), ("XLI", "Industrials"), ("IWM", "Small_Caps")):
        snap = yf_snapshot(symbol)
        if snap:
            sector_momentum[label] = snap["momentum_3m_pct"]

    return {
        "treasury_10y_pct": treasury_now,
        "treasury_10y_pct_6mo_ago": treasury_6m_ago,
        "treasury_10y_trend": treasury_trend,
        "breakeven_inflation_pct": inflation_now,
        "breakeven_inflation_pct_6mo_ago": inflation_6m_ago,
        "breakeven_inflation_trend": inflation_trend,
        "sp500_vs_200sma_pct": sp500_vs_200sma_pct,
        "sector_momentum_3m_pct": sector_momentum,
    }


# -- source 5: requests + BeautifulSoup (Morningstar) ------------------------

def morningstar_headlines(limit: int = 5) -> list[str]:
    """Best-effort qualitative signal only. The free Morningstar ETFs page
    is an editorial hub (article headlines), not a structured ratings
    table — this is NOT a ranked "top ETFs" list, just recent headline
    text for color. ETF.com is Cloudflare-gated and not scraped."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; portfolio-rebalancer/1.0)"}
        resp = requests.get("https://www.morningstar.com/etfs", headers=headers, timeout=10)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "lxml")
        headlines = []
        for tag in soup.find_all(["h2", "h3", "a"]):
            text = tag.get_text(strip=True)
            if 15 < len(text) < 120 and re.search(r"ETF|Fund", text, re.IGNORECASE):
                if text not in headlines:
                    headlines.append(text)
            if len(headlines) >= limit:
                break
        return headlines
    except Exception:
        return []


# -- pro-trader analytics (computed in Python, never trusted from the LLM) --

def compute_overlap_pct(candidate_holdings: list[str], portfolio_holdings_union: set[str]) -> float:
    """% of the candidate's own top-10 holdings that already appear in the
    combined top-10 holdings of the account's current ETF positions."""
    if not candidate_holdings:
        return 0.0
    overlap = sum(1 for h in candidate_holdings if h in portfolio_holdings_union)
    return round(overlap / len(candidate_holdings) * 100, 1)


def snapshot_held_tickers(held_tickers: list[str]) -> dict[str, dict]:
    """Full yfinance snapshot per held ticker, keyed by symbol — the
    single fetch reused for both the holdings-overlap union AND (for
    modes 5/6/7) each held position's technical/forward metrics, so
    nothing gets fetched twice."""
    out = {}
    for ticker in held_tickers:
        snap = yf_snapshot(ticker)
        if snap:
            out[ticker] = snap
    return out


_TECHNICAL_FIELDS = ("rsi_14", "rsi_signal", "sma50_deviation_pct", "sma200_deviation_pct", "bollinger_signal", "trailing_pe")


def build_market_intelligence(
    held_tickers: list[str],
    risk_tolerance: int | None = None,
    extra_candidates: list[str] | None = None,
    include_forward: bool = False,
    include_macro: bool = False,
) -> dict:
    """Orchestrates all 5 sources into one compact, Claude-ready dict.
    Every candidate ETF returned has been fetched for real via yfinance
    (so it's a tradable, real ticker) and has already passed the expense
    ratio (<=0.50%) and overlap (<=60%) hard filters — candidates that
    fail are returned separately under rejected_candidates for
    transparency, not silently dropped.

    include_forward (Mode 5) adds a look-through forward P/E / earnings
    growth / implied 12mo return to every held + candidate entry — this
    costs extra network round-trips (up to top_n stock lookups per ETF,
    cached per-request) so it's opt-in, not default.
    include_macro (Mode 6) adds a macro_regime block (Treasury/inflation
    trend, sector momentum) — cheap (a handful of calls total), still
    opt-in since it's only relevant to that mode."""
    held = [h.upper() for h in held_tickers]
    held_snapshots = snapshot_held_tickers(held)
    portfolio_holdings_union = set()
    for snap in held_snapshots.values():
        portfolio_holdings_union.update(snap.get("top_holdings") or [])

    benchmark = sp500_benchmark_momentum()
    forward_cache: dict[str, dict] = {}

    risk_tolerance = 50 if risk_tolerance is None else max(1, min(100, risk_tolerance))
    bands = ["conservative"]
    if risk_tolerance > 25:
        bands.append("moderate")
    if risk_tolerance > 55:
        bands.append("aggressive")
    if risk_tolerance >= 80:
        bands.append("very_aggressive")
    risk_universe = [t for band in bands for t in CANDIDATE_UNIVERSE[band]]

    finviz_raw = finviz_momentum_leaders(limit=5)
    raw_candidates = list(dict.fromkeys(risk_universe + finviz_raw + (extra_candidates or [])))
    raw_candidates = [t for t in raw_candidates if t not in held]

    accepted, rejected = [], []
    for ticker in raw_candidates:
        snap = yf_snapshot(ticker)
        if not snap:
            rejected.append({"ticker": ticker, "reason": "not a resolvable/tradable ticker (yfinance lookup failed)"})
            continue

        expense_ratio = snap["expense_ratio_pct"]
        if expense_ratio is None:
            fallback = stockdex_expense_ratio(ticker)
            if fallback is not None:
                expense_ratio = fallback
        if expense_ratio is not None and expense_ratio > EXPENSE_RATIO_MAX_PCT:
            rejected.append({"ticker": ticker, "reason": f"expense ratio {expense_ratio}% exceeds {EXPENSE_RATIO_MAX_PCT}% cap"})
            continue

        overlap_pct = compute_overlap_pct(snap["top_holdings"], portfolio_holdings_union)
        if overlap_pct > OVERLAP_MAX_PCT:
            rejected.append({"ticker": ticker, "reason": f"{overlap_pct}% holdings overlap with current portfolio exceeds {OVERLAP_MAX_PCT}% cap"})
            continue

        relative_strength_3m = None
        if snap["momentum_3m_pct"] is not None and benchmark["momentum_3m_pct"] is not None:
            relative_strength_3m = round(snap["momentum_3m_pct"] - benchmark["momentum_3m_pct"], 2)

        entry = {
            "ticker": ticker,
            "name": snap["name"],
            "category": snap["category"],
            "last_price": snap["last_price"],
            "expense_ratio_pct": expense_ratio,
            "momentum_3m_pct": snap["momentum_3m_pct"],
            "momentum_6m_pct": snap["momentum_6m_pct"],
            "relative_strength_vs_sp500_3m_pct": relative_strength_3m,
            "overlap_with_portfolio_pct": overlap_pct,
            **{k: snap[k] for k in _TECHNICAL_FIELDS},
        }
        if include_forward:
            entry.update(yf_forward_lookthrough(snap.get("top_holdings_weighted") or [], forward_cache))
        accepted.append(entry)

    held_metrics = []
    for ticker, snap in held_snapshots.items():
        entry = {
            "ticker": ticker,
            "momentum_3m_pct": snap["momentum_3m_pct"],
            "momentum_6m_pct": snap["momentum_6m_pct"],
            **{k: snap[k] for k in _TECHNICAL_FIELDS},
        }
        if include_forward:
            entry.update(yf_forward_lookthrough(snap.get("top_holdings_weighted") or [], forward_cache))
        held_metrics.append(entry)

    headlines = morningstar_headlines(limit=5)
    result = {
        "sp500_benchmark": benchmark,
        "candidates": accepted[:10],
        "rejected_candidates": rejected[:10],
        "held_metrics": held_metrics,
        "macro_regime": macro_regime_signals() if include_macro else None,
        "headlines": headlines,
        "sources_used": {
            "yfinance": True,
            "fred_macro": benchmark["momentum_3m_pct"] is not None,
            "finviz_momentum": len(finviz_raw) > 0,
            "stockdex_justetf": any(c.get("expense_ratio_pct") is not None for c in accepted) and StockdexTicker is not None,
            "morningstar_headlines": len(headlines) > 0,
        },
    }
    _debug("market intelligence summary", result)
    return result
