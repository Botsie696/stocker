"""Shared analyze/execute logic for both the Flask dashboard (app.py) and
the headless CLI entrypoint (run_headless.py) used by GitHub Actions.

This module exists so there is exactly ONE implementation of the
safety-critical path — guardrail validation, mode-specific checks
(the forward-return friction guardrail, Mode 7 pairing), and the strict
sell-then-buy execution sequence with live full-liquidation resolution and
dynamic buying-power capping. Duplicating this logic between an
interactive route and a CLI script is how a guardrail quietly stops
applying in one of the two places; importing the same functions from both
entrypoints makes that impossible by construction.

Nothing here talks HTTP — analyze() and execute_trades() take and return
plain dicts/generators. app.py wraps them for Flask (jsonify, NDJSON
streaming); run_headless.py wraps them for stdout logging.

Scheduling is entirely cloud-native (github_service.py's Contents API
queue) — there is no local file-based pending-batch mechanism here.
serialize_batch() below is what turns an in-memory batch into the JSON
committed to queue/scheduled_task.json.
"""
import os

import data_scraper
import guardrails
from claude_engine import get_rebalance_proposal
from public_client import PublicAPIError, PublicClient, parse_portfolio

VALID_MODES = {"1", "2", "3", "4", "5", "6", "7", "8"}
VALID_SUB_MODES = {"4A", "4B"}
FORWARD_FRICTION_THRESHOLD_PCT = 2.5
FORWARD_MODES = {"5", "8"}  # modes that fetch include_forward=True market intelligence
FAILED_TRADE_STATUSES = {"REJECTED_GUARDRAIL", "FAILED_PREFLIGHT", "FAILED_ORDER", "CANCELLED", "REJECTED", "EXPIRED", "QUEUED_CANCELLED"}


class BadRequest(ValueError):
    pass


def get_client() -> PublicClient:
    secret = os.environ.get("PUBLIC_API_SECRET")
    if not secret:
        raise RuntimeError("PUBLIC_API_SECRET is not set (see .env.example)")
    account_id = os.environ.get("PUBLIC_ACCOUNT_ID") or None
    return PublicClient(secret=secret, account_id=account_id)


def filter_and_weight(all_positions: list[dict], whitelist: set[str], total_value: float) -> list[dict]:
    """Reduces normalized positions to whitelisted ETFs only (Claude never
    sees non-whitelisted holdings as sell candidates), and attaches each
    position's weight as a percentage of the FULL portfolio total_value
    (which includes non-whitelisted holdings + cash) — not just the
    whitelisted subset — so the weight numbers reflect true concentration."""
    out = []
    for p in all_positions:
        if p["ticker"] not in whitelist:
            continue
        weight_pct = (p["market_value"] / total_value * 100) if total_value else 0.0
        out.append({**p, "weight_pct": round(weight_pct, 2)})
    return out


def resolve_full_liquidations(raw_trades: list[dict], positions: list[dict]) -> list[dict]:
    """SELL trades flagged is_full_liquidation get their amount_usd
    recomputed from the position's real, live market value at proposal
    time — Claude's own dollar guess is discarded, consistent with never
    trusting the LLM's own math. execute_trades() re-resolves this AGAIN
    right before the order actually fires, in case the position's value
    moved between proposal and execution (a cloud-scheduled batch can fire
    hours or a day later)."""
    value_by_ticker = {p["ticker"]: p["market_value"] for p in positions}
    out = []
    for t in raw_trades:
        t = dict(t)
        if t.get("action") == "SELL" and t.get("is_full_liquidation"):
            market_value = value_by_ticker.get(str(t.get("ticker", "")).upper())
            if market_value:
                t["amount_usd"] = round(market_value, 2)
        out.append(t)
    return out


def lookup_implied_return(ticker: str, market_intelligence: dict) -> float | None:
    for c in market_intelligence.get("candidates", []):
        if c["ticker"] == ticker:
            return c.get("implied_12m_return_pct")
    for h in market_intelligence.get("held_metrics", []):
        if h["ticker"] == ticker:
            return h.get("implied_12m_return_pct")
    return None


def apply_forward_friction_guardrail(
    trades: list[dict], market_intelligence: dict, threshold: float = FORWARD_FRICTION_THRESHOLD_PCT,
) -> tuple[list[dict], str | None]:
    """Hardcoded backstop for the "must beat current holdings by >=2.5pp
    or HOLD" rule — enforced here in Python, not just requested in the
    prompt. Applies to modes 5 and 8, the only ones that fetch look-through
    forward-return data (implied_12m_return_pct); other modes get this as
    prompt-level guidance only since Python has no independent "expected
    annual return" figure to check it against for those strategies (e.g.
    Mode 1's relative-performance framing or Mode 6's macro-regime
    reasoning don't reduce to a single comparable return number).
    Compares the dollar-weighted average implied 12-month return of all
    proposed SELLs against all proposed BUYs; if the improvement doesn't
    clear the bar (or the underlying forward data is missing), the whole
    batch is discarded in favor of a HOLD rather than guessing which
    individual pair was the problem."""
    sells = [t for t in trades if t["action"] == "SELL"]
    buys = [t for t in trades if t["action"] == "BUY"]
    if not sells or not buys:
        return trades, None

    def weighted_avg(trade_list):
        total = sum(t["amount_usd"] for t in trade_list) or 1.0
        acc = 0.0
        for t in trade_list:
            r = lookup_implied_return(t["ticker"], market_intelligence)
            if r is None:
                return None
            acc += r * t["amount_usd"]
        return acc / total

    sell_avg = weighted_avg(sells)
    buy_avg = weighted_avg(buys)
    if sell_avg is None or buy_avg is None:
        return trades, None  # insufficient forward data to enforce; don't block on it

    improvement = buy_avg - sell_avg
    if improvement < threshold:
        return [], (
            f"Friction guardrail: proposed trades' projected 12mo return improvement "
            f"({improvement:.2f}pp) is below the required {threshold}pp minimum — "
            f"held as HOLD instead of trading."
        )
    return trades, None


def apply_mode7_pairing_guardrail(trades: list[dict]) -> tuple[list[dict], str | None]:
    """Mode 7's spec is explicitly a pair: "take profits from overextended
    holdings AND deploy that cash into oversold ETFs" — not standalone
    profit-taking. If there's no oversold BUY candidate, there's nothing
    to mean-revert into, so the whole batch is discarded in favor of a
    HOLD rather than taking profits into idle cash."""
    sells = [t for t in trades if t["action"] == "SELL"]
    buys = [t for t in trades if t["action"] == "BUY"]
    if sells and not buys:
        return [], (
            "Pairing guardrail: overbought holdings were identified, but no oversold "
            "candidate qualified to redeploy the proceeds into — held as HOLD rather "
            "than taking profits into idle cash."
        )
    return trades, None


def parse_analyze_request(body: dict, whitelist: set[str], held_symbols: set[str]) -> dict:
    mode = str(body.get("mode", "1")).strip()
    if mode not in VALID_MODES:
        raise BadRequest(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}")

    sub_mode = body.get("sub_mode")
    if mode == "4":
        sub_mode = str(sub_mode or "").strip().upper()
        if sub_mode not in VALID_SUB_MODES:
            raise BadRequest(f"mode 4 requires sub_mode one of {sorted(VALID_SUB_MODES)}")
    else:
        sub_mode = None

    budget_usd = body.get("budget_usd")
    if budget_usd not in (None, ""):
        try:
            budget_usd = float(budget_usd)
        except (TypeError, ValueError):
            raise BadRequest(f"budget_usd must be numeric, got {budget_usd!r}")
        if budget_usd <= 0:
            raise BadRequest("budget_usd must be positive")
    else:
        budget_usd = None

    needs_budget = mode == "3" or (mode == "4" and sub_mode == "4B")
    if needs_budget and budget_usd is None:
        raise BadRequest("this mode requires a Budget amount")

    risk_tolerance = body.get("risk_tolerance", 50)
    try:
        risk_tolerance = int(risk_tolerance)
    except (TypeError, ValueError):
        raise BadRequest(f"risk_tolerance must be an integer 1-100, got {risk_tolerance!r}")
    risk_tolerance = max(1, min(100, risk_tolerance))

    custom_instructions = (body.get("custom_instructions") or "").strip() or None

    max_trade_usd = budget_usd if budget_usd is not None else guardrails.get_max_trade_usd()

    market_intelligence = {}
    if mode in ("2", "4", "5", "6", "7", "8"):
        market_intelligence = data_scraper.build_market_intelligence(
            held_tickers=list(held_symbols),
            # Mode 8 isn't risk-banded — it's "does this mathematically
            # clear the hurdle rate," not a risk-appetite dial — so it
            # always gets the broadest candidate universe (all bands)
            # rather than whatever the UI slider happens to be set to.
            risk_tolerance=100 if mode == "8" else risk_tolerance,
            include_forward=(mode in FORWARD_MODES),
            include_macro=(mode in ("6", "8")),
        )
    candidate_tickers = {c["ticker"] for c in market_intelligence.get("candidates", [])}

    if mode == "1":
        allowed_tickers = set(held_symbols)
    elif mode == "2":
        allowed_tickers = set(held_symbols) | candidate_tickers
    elif mode == "3":
        allowed_tickers = set(whitelist)
    elif mode == "4":
        allowed_tickers = (set(held_symbols) if sub_mode == "4A" else set(whitelist)) | candidate_tickers
    else:  # modes 5, 6, 7, 8 — sell only from held, buy from held+candidates
        allowed_tickers = set(held_symbols) | candidate_tickers

    return {
        "mode": mode,
        "sub_mode": sub_mode,
        "budget_usd": budget_usd,
        "risk_tolerance": risk_tolerance,
        "custom_instructions": custom_instructions,
        "max_trade_usd": max_trade_usd,
        "market_intelligence": market_intelligence,
        "allowed_tickers": allowed_tickers,
    }


def analyze(body: dict, client: PublicClient | None = None) -> dict:
    """Full propose-only flow: fetch fresh portfolio state, build the
    mode-appropriate candidate universe, ask Claude for a proposal, and
    hard-filter the response through guardrails. Nothing is executed here.
    Returns a plain dict — the caller (Flask route or CLI script) decides
    how to store/display/serialize it."""
    client = client or get_client()
    normalized = parse_portfolio(client.get_portfolio())
    whitelist = guardrails.get_whitelist()
    positions = filter_and_weight(normalized["positions"], whitelist, normalized["total_value"])
    cash = normalized["cash"]
    total_value = normalized["total_value"]
    held_symbols = {p["ticker"] for p in positions}

    params = parse_analyze_request(body, whitelist, held_symbols)

    proposal = get_rebalance_proposal(
        mode=params["mode"],
        cash=cash,
        total_value=total_value,
        positions=positions,
        allowed_tickers=params["allowed_tickers"],
        max_trade_usd=params["max_trade_usd"],
        sub_mode=params["sub_mode"],
        budget_usd=params["budget_usd"],
        risk_tolerance=params["risk_tolerance"],
        custom_instructions=params["custom_instructions"],
        market_intelligence=params["market_intelligence"],
    )

    # action="HOLD" is authoritative over whatever's in trades — if Claude
    # says HOLD but still listed trades (shouldn't happen, but never trust
    # the LLM to be internally consistent either), treat it as no trades.
    raw_trades = [] if proposal.get("action") == "HOLD" else proposal.get("trades", [])
    raw_trades = resolve_full_liquidations(raw_trades, positions)

    accepted, rejected = guardrails.validate_batch(
        raw_trades, held_symbols,
        allowed_tickers=params["allowed_tickers"],
        max_trade_usd=params["max_trade_usd"],
        budget_usd=params["budget_usd"],
    )

    rationale = proposal.get("overall_rationale", "")
    if params["mode"] in FORWARD_MODES and accepted:
        accepted, hold_reason = apply_forward_friction_guardrail(accepted, params["market_intelligence"])
        if hold_reason:
            rationale = f"{rationale}\n\n[System] {hold_reason}"
    if params["mode"] == "7" and accepted:
        accepted, hold_reason = apply_mode7_pairing_guardrail(accepted)
        if hold_reason:
            rationale = f"{rationale}\n\n[System] {hold_reason}"

    price_by_ticker = {p["ticker"]: p["last_price"] for p in positions}
    for c in params["market_intelligence"].get("candidates", []):
        price_by_ticker.setdefault(c["ticker"], c["last_price"])
    missing = [t["ticker"] for t in accepted if t["ticker"] not in price_by_ticker]
    if missing:
        fetched = client.get_quotes(missing)
        for sym, q in fetched.items():
            price_by_ticker[sym] = q.last

    for t in accepted:
        price = price_by_ticker.get(t["ticker"]) or 0
        t["estimated_shares"] = round(t["amount_usd"] / price, 4) if price else None

    return {
        "rationale": rationale,
        "trades": accepted,
        "rejected": rejected,
        "cash": cash,
        "total_value": total_value,
        "allowed_tickers": params["allowed_tickers"],
        "max_trade_usd": params["max_trade_usd"],
        "budget_usd": params["budget_usd"],
        "mode": params["mode"],
        "sub_mode": params["sub_mode"],
        "market_intelligence": params["market_intelligence"],
    }


def execute_trades(
    client: PublicClient,
    trades: list[dict],
    allowed_tickers: set[str],
    max_trade_usd: float,
    budget_usd: float | None,
    dry_run: bool,
):
    """Strict 2-phase gated execution. A generator yielding plain event
    dicts ({"type": "info"|"trade_result"|"done"|"error", ...}) — callers
    render them however fits (NDJSON stream for the Flask route, stdout
    log lines for the CLI script).
      Phase 1 SELL — fire all SELL orders first, sequentially. Any SELL
                     flagged is_full_liquidation gets its dollar amount
                     re-resolved from the LIVE current position value
                     right here (not whatever was approved earlier) — the
                     position may have moved since (a cloud-scheduled
                     batch can fire a day later), and a stale amount could
                     leave a small residual instead of a true full exit.
      Phase 2 BUY  — no settlement-wait loop. Public.com grants buying
                     power the instant a SELL fills, even before cash
                     formally settles T+1 — and on a margin account,
                     buying power can already be sufficient *before* the
                     SELL even executes, so waiting for it to "increase"
                     is unreliable either way. Instead: check current
                     buying power ONCE (skipped entirely if nothing was
                     sold), then cap each BUY sequentially to whatever's
                     actually available rather than an all-or-nothing
                     abort — a BUY that only partially fits still gets as
                     much of itself filled as the cash allows.
    If a SELL outright fails, all BUY orders are aborted (nothing to fund
    them safely). Every trade is re-validated against guardrails using the
    exact allowed-ticker set and dollar caps the caller passes in — the
    batch may be stale by the time execution actually starts."""
    log = []
    normalized = parse_portfolio(client.get_portfolio())
    whitelist = guardrails.get_whitelist()
    live_positions = filter_and_weight(normalized["positions"], whitelist, normalized["total_value"])
    held_symbols = {p["ticker"] for p in live_positions}
    position_value_by_ticker = {p["ticker"]: p["market_value"] for p in live_positions}

    validated_trades, revalidation_rejected = guardrails.validate_batch(
        trades, held_symbols,
        allowed_tickers=set(allowed_tickers),
        max_trade_usd=max_trade_usd,
        budget_usd=budget_usd,
    )
    for r in revalidation_rejected:
        entry = {**r, "status": "REJECTED_GUARDRAIL", "detail": r.get("rejection_reason", "failed re-validation at execute time")}
        log.append(entry)
        yield {"type": "trade_result", **entry}

    if not validated_trades:
        yield {"type": "info", "message": "No trades to execute."}
        yield {"type": "done", "log": log}
        return

    def run_trade(t):
        if dry_run:
            entry = {**t, "status": "DRY_RUN", "detail": "simulated — no order submitted"}
            log.append(entry)
            return entry
        try:
            client.preflight_order(t["ticker"], t["action"], t["amount_usd"])
        except PublicAPIError as e:
            entry = {**t, "status": "FAILED_PREFLIGHT", "detail": str(e)}
            log.append(entry)
            return entry
        try:
            order_id = client.place_order(t["ticker"], t["action"], t["amount_usd"])
            final = client.wait_for_fill(order_id, timeout_s=60, poll_s=2)
            entry = {
                **t,
                "status": final.get("status", "UNKNOWN"),
                "order_id": order_id,
                "detail": f"filled_qty={final.get('filledQuantity')} avg_price={final.get('averagePrice')}",
            }
        except PublicAPIError as e:
            entry = {**t, "status": "FAILED_ORDER", "detail": str(e)}
        log.append(entry)
        return entry

    sells = [t for t in validated_trades if t["action"] == "SELL"]
    buys = [t for t in validated_trades if t["action"] == "BUY"]

    # Phase 1: SELL
    if sells:
        yield {"type": "info", "message": f"Phase 1/2 (SELL) — placing {len(sells)} order(s)..."}
    sell_failed = False
    for t in sells:
        if t.get("is_full_liquidation"):
            live_value = position_value_by_ticker.get(t["ticker"])
            if live_value and round(live_value, 2) != t["amount_usd"]:
                yield {"type": "info", "message": f"Full liquidation of {t['ticker']}: using live position value ${live_value:,.2f} (was ${t['amount_usd']:,.2f} at approval time)"}
            if live_value:
                t = {**t, "amount_usd": round(live_value, 2)}
        verb = "Simulating" if dry_run else "Submitting"
        yield {"type": "info", "message": f"{verb} SELL ${t['amount_usd']:.2f} of {t['ticker']}..."}
        entry = run_trade(t)
        yield {"type": "trade_result", **entry}
        if entry["status"] in FAILED_TRADE_STATUSES:
            sell_failed = True

    if sell_failed and buys:
        yield {"type": "info", "message": "A SELL order failed — aborting all BUY orders."}
        for b in buys:
            entry = {**b, "status": "ABORTED_SELL_FAILED", "detail": "a required SELL did not fill; BUY orders aborted"}
            log.append(entry)
            yield {"type": "trade_result", **entry}
        yield {"type": "done", "log": log, "warning": "one or more SELL orders failed; all BUY orders aborted"}
        return

    # Phase 2: BUY
    if buys:
        yield {"type": "info", "message": f"Phase 2/2 (BUY) — placing {len(buys)} order(s)..."}
        remaining_buying_power = None
        if not dry_run:
            remaining_buying_power = client.get_buying_power()
            yield {"type": "info", "message": f"Live buying power: ${remaining_buying_power:,.2f}"}

        for t in buys:
            if remaining_buying_power is not None:
                capped = round(min(t["amount_usd"], max(remaining_buying_power, 0)), 2)
                if capped <= 0:
                    entry = {**t, "status": "SKIPPED_INSUFFICIENT_BUYING_POWER", "detail": f"no buying power remaining (needed ${t['amount_usd']:.2f})"}
                    log.append(entry)
                    yield {"type": "trade_result", **entry}
                    continue
                if capped < t["amount_usd"]:
                    yield {"type": "info", "message": f"Capping {t['ticker']} BUY to ${capped:.2f} (requested ${t['amount_usd']:.2f}, live buying power limit)"}
                t = {**t, "amount_usd": capped}
                remaining_buying_power -= capped
            verb = "Simulating" if dry_run else "Submitting"
            yield {"type": "info", "message": f"{verb} BUY ${t['amount_usd']:.2f} of {t['ticker']}..."}
            entry = run_trade(t)
            yield {"type": "trade_result", **entry}

    yield {"type": "done", "log": log}


# -- pre-approved batch: skip Claude entirely, execute a plan a human ------
# -- already reviewed, via the cloud queue (github_service.py's Contents  --
# -- API — see queue/scheduled_task.json) or an immediate GitHub dispatch --

def serialize_batch(batch: dict) -> dict:
    """JSON-safe view of a batch dict (allowed_tickers is a set in memory,
    which json.dumps chokes on) — this is the shape committed to
    queue/scheduled_task.json and passed to GitHub Actions."""
    return {
        "trades": batch["trades"],
        "allowed_tickers": sorted(batch["allowed_tickers"]),
        "max_trade_usd": batch["max_trade_usd"],
        "budget_usd": batch["budget_usd"],
    }


def execute_approved_batch(batch: dict, dry_run: bool):
    """Unified entrypoint for executing a PRE-APPROVED batch — one already
    reviewed by a human and validated once at proposal time, arriving here
    from either the cloud queue file or an immediate trades_json dispatch.
    Claude is never called. guardrails.validate_batch() still re-runs
    inside execute_trades() as a defense-in-depth check (the batch may be
    stale by the time this actually fires), and the market-hours gate is
    enforced here so every caller gets it without duplicating the check.
    Yields the same event dicts as execute_trades()."""
    if not dry_run and not guardrails.is_market_hours():
        yield {"type": "error", "message": "market is closed (Mon-Fri 9:30am-4:00pm America/New_York); execution skipped"}
        yield {"type": "done", "log": [], "warning": "market was closed when this pre-approved batch fired"}
        return
    client = get_client()
    yield from execute_trades(
        client, batch["trades"], set(batch["allowed_tickers"]), batch["max_trade_usd"], batch["budget_usd"], dry_run,
    )
