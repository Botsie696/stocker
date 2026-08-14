"""Hardcoded safety checks. Claude's output is never trusted directly —
every proposed trade is re-validated here before it can reach the broker,
both when proposals are generated AND again immediately before execution.
"""
import os
from datetime import datetime
from zoneinfo import ZoneInfo

MARKET_TZ = ZoneInfo("America/New_York")


def get_whitelist() -> set[str]:
    raw = os.environ.get("ETF_WHITELIST", "VOO,QQQM,AVUV,CALF,AVLV,RZV,RWJ")
    return {t.strip().upper() for t in raw.split(",") if t.strip()}


def get_max_trade_usd() -> float:
    return float(os.environ.get("MAX_TRADE_USD", "500"))


def is_market_hours(now: datetime | None = None) -> bool:
    """Mon-Fri, 9:30am-4:00pm America/New_York. Does NOT account for market
    holidays (e.g. Thanksgiving, Christmas) — this is a real gap, not an
    oversight; add a market-calendar check (e.g. pandas_market_calendars)
    before relying on this alone for holiday sessions."""
    now = now.astimezone(MARKET_TZ) if now else datetime.now(MARKET_TZ)
    if now.weekday() >= 5:  # Sat/Sun
        return False
    open_t = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_t <= now <= close_t


def validate_trade(
    trade: dict,
    held_symbols: set[str],
    allowed_tickers: set[str] | None = None,
    max_trade_usd: float | None = None,
) -> tuple[bool, str]:
    """Validate a single proposed trade dict: {action, ticker, amount_usd}.
    Returns (is_valid, reason_if_rejected).

    allowed_tickers defaults to the static ETF_WHITELIST (Modes 1 & 3).
    For Modes 2/4 the caller passes a wider set — the static whitelist
    UNION whatever external candidates data_scraper.py already vetted
    (real ticker, expense ratio <=0.50%, overlap <=60%) for this run.
    Claude can only ever land a ticker that was pre-approved by Python
    before this call — never one it invents.

    max_trade_usd defaults to the static MAX_TRADE_USD env ceiling; the
    UI's Budget field can override it per-request (see app.py), but that
    override is still a Python-enforced number decided before the batch
    was proposed, not something Claude's own output can raise.
    """
    allowed_tickers = get_whitelist() if allowed_tickers is None else allowed_tickers
    max_usd = get_max_trade_usd() if max_trade_usd is None else max_trade_usd

    action = str(trade.get("action", "")).upper()
    ticker = str(trade.get("ticker", "")).upper()
    amount_usd = trade.get("amount_usd")

    if action not in ("BUY", "SELL"):
        return False, f"action must be BUY or SELL, got {action!r}"

    if ticker not in allowed_tickers:
        return False, f"{ticker} is not in the approved ticker set for this request ({sorted(allowed_tickers)})"

    if action == "SELL" and ticker not in held_symbols:
        return False, f"cannot SELL {ticker}: not currently held in the account"

    if not isinstance(amount_usd, (int, float)) or amount_usd <= 0:
        return False, f"amount_usd must be a positive number, got {amount_usd!r}"

    if amount_usd > max_usd:
        return False, f"amount_usd {amount_usd} exceeds max order ceiling ${max_usd}"

    return True, ""


def validate_batch(
    trades: list[dict],
    held_symbols: set[str],
    allowed_tickers: set[str] | None = None,
    max_trade_usd: float | None = None,
    budget_usd: float | None = None,
) -> tuple[list[dict], list[dict]]:
    """Split proposed trades into (accepted, rejected_with_reason).

    When budget_usd is given (the UI's Budget field), it's also enforced
    as a hard cumulative cap across all BUY trades in the batch — not just
    a per-trade ceiling. Trades are kept in the order Claude proposed them
    until the running BUY total would exceed budget_usd; anything past
    that point is rejected rather than silently shrunk."""
    accepted, rejected = [], []
    for t in trades:
        ok, reason = validate_trade(t, held_symbols, allowed_tickers=allowed_tickers, max_trade_usd=max_trade_usd)
        if ok:
            accepted.append(t)
        else:
            rejected.append({**t, "rejection_reason": reason})

    if budget_usd is not None:
        running_total = 0.0
        kept = []
        for t in accepted:
            if t["action"] == "BUY":
                if running_total + t["amount_usd"] > budget_usd + 1e-6:
                    rejected.append({**t, "rejection_reason": f"cumulative BUY total would exceed budget ${budget_usd:,.2f}"})
                    continue
                running_total += t["amount_usd"]
            kept.append(t)
        accepted = kept

    return accepted, rejected
