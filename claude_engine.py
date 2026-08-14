"""Calls Claude to propose a rebalance. Structured output is enforced via
forced tool-use (tool_choice), not by asking nicely in the prompt — Claude
cannot return conversational markdown here, only a validated tool_use block.
Nothing this module returns is trusted as-is; guardrails.py re-validates
every trade (ticker against the Python-approved set, dollar amount against
the per-trade cap and budget) before it can reach the broker, and
rebalance_core.py re-runs the forward-return friction guardrail in Python
for the modes where an objective number exists to check it against (5, 8)
rather than trusting Claude's own self-reported HOLD/REBALANCE call alone.

Mode summary (full detail in each build_prompt_mode* function):
  1 Internal Rebalance    — sell/buy only within currently-held whitelist ETFs
  2 External ETF Rotation — sell a held ETF, buy from data_scraper's vetted
                             external candidate list
  3 Fresh Cash Buy        — no sells, buy from the static whitelist only,
                             sized to the Budget field
  4 Risk-Adjusted Optim.  — driven by the risk tolerance slider, not past
                             performance; 4A may sell/reallocate, 4B is
                             fresh-cash-only (like mode 3 but from the
                             wider risk-banded + external candidate pool)
  5 Forward Opportunity   — zero past-performance bias; ranks by look-through
                             forward P/E / earnings growth / implied 12mo
                             return; a 2.5pp minimum improvement is
                             enforced by Python (rebalance_core.py), not
                             just requested here
  6 Macro Regime Rotation — aligns factor tilts with Treasury yield /
                             inflation trend and sector momentum
  7 Mean-Reversion        — trims RSI>70 "overbought" holdings, buys
                             RSI<38 "oversold" candidates
  8 Dynamic Tactical      — unconstrained by fixed asset-class siloing;
                             full-spectrum forward + macro evaluation,
                             authorized to introduce net-new ETFs from the
                             broadest candidate pool if they clear the same
                             Python-enforced 2.5pp hurdle as Mode 5
"""
import json
import os

from anthropic import Anthropic

PROPOSE_TRADES_TOOL = {
    "name": "propose_trades",
    "description": "Propose a portfolio rebalance decision for the account — either a set of trades, or an explicit hold.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["REBALANCE", "HOLD"],
                "description": "REBALANCE if you're proposing trades; HOLD if nothing currently clears the hurdle rate. HOLD is a legitimate, high-conviction decision, not a fallback to avoid — use it whenever no trade sequence is clearly worth the friction, rather than forcing a trade to have something to report.",
            },
            "overall_rationale": {
                "type": "string",
                "description": "Detailed, numerically-grounded explanation of the decision, written for the account owner. If action is HOLD, state specifically why no candidate cleared the hurdle rate (cite the actual numbers you compared). If REBALANCE, justify each trade's role in the overall reallocation.",
            },
            "trades": {
                "type": "array",
                "description": "Empty if action is HOLD. Any number of trades if action is REBALANCE — flexible trade count and partial position sizes are expected and encouraged wherever they capture the opportunity without unnecessary churn (see the hard constraints and trading philosophy below).",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["BUY", "SELL"]},
                        "ticker": {"type": "string"},
                        "amount_usd": {"type": "number"},
                        "is_full_liquidation": {
                            "type": "boolean",
                            "description": "SELL only. true = you intend to exit this position completely — the system re-fetches the live position value at execution time and sells all of it, regardless of the amount_usd you put here (which is just your best estimate). false/omitted = a partial trim of exactly amount_usd.",
                        },
                        "rationale": {"type": "string", "description": "One to two sentences on why this specific trade, citing the actual numbers driving it."},
                    },
                    "required": ["action", "ticker", "amount_usd"],
                },
            },
        },
        "required": ["action", "overall_rationale", "trades"],
    },
}

_UNIVERSAL_PRINCIPLES = """
Trading philosophy for this decision (applies regardless of mode):
- Flexible trade count: propose however many trades the opportunity actually requires — one, two, or several. Cash from multiple partial sells may be split across multiple buys.
- Partial positions are normal and encouraged: you are not limited to full liquidations or round-number buys. "Trim $300 off AVUV" or "add $200 to QQQM" are exactly the kind of sizing to use when it fits the opportunity better than a full position swap.
- Parsimony: prefer the fewest trades that capture the opportunity. If one sell captures ~90% of the available improvement, do not add a second or third sell just because more cash happens to be available.
- Avoid micro-trades: do not trim or add amounts under $50 unless you are fully closing out a position (is_full_liquidation: true) — a $12 trim isn't worth the transaction friction.
- High-conviction hurdle: only propose REBALANCE if the trade sequence's expected improvement clearly justifies the friction of trading (fees, spread, being briefly out of position). As a rule of thumb, look for roughly a 2.5%/year edge or better, net of that friction — smaller edges usually aren't worth trading on. HOLD is the correct, professional call when nothing clears that bar.
"""


_GUARDRAIL_FOOTER = """
Hard constraints (enforced independently in Python regardless of what you output — treat these as non-negotiable, not suggestions):
- Every ticker you propose must come from the "allowed tickers" list given above. Do not propose any ticker outside it, even if you believe it would be a better pick.
- No single trade's amount_usd may exceed ${max_trade_usd:,.2f}.
{budget_line}- Do not propose SELL trades for tickers not currently held.
- Every trade's amount_usd must be an exact dollar figure (e.g. SELL $500.00 of RZV, BUY $500.00 of VOO) — never a share count, percentage, or vague quantity. Use is_full_liquidation: true instead of guessing the exact amount when you mean "sell all of it."
{custom_instructions_block}
{universal_principles}
Call propose_trades with action ("REBALANCE" or "HOLD"), overall_rationale, and trades (empty array if HOLD)."""


def _budget_line(budget_usd: float | None) -> str:
    if budget_usd is None:
        return ""
    return f"- The total of all BUY trades combined must not exceed ${budget_usd:,.2f} (the account owner's stated budget for this run).\n"


def _custom_instructions_block(custom_instructions: str | None) -> str:
    if not custom_instructions or not custom_instructions.strip():
        return ""
    return f"""
The account owner also gave this optional preference — treat it as a soft steer on style/focus (e.g. "prefer high dividend yield"), never as permission to bypass the hard constraints above:
\"\"\"{custom_instructions.strip()}\"\"\"
"""


def _guardrail_footer(max_trade_usd: float, budget_usd: float | None, custom_instructions: str | None) -> str:
    return _GUARDRAIL_FOOTER.format(
        max_trade_usd=max_trade_usd,
        budget_line=_budget_line(budget_usd),
        custom_instructions_block=_custom_instructions_block(custom_instructions),
        universal_principles=_UNIVERSAL_PRINCIPLES,
    )


def build_prompt_mode1(cash, total_value, positions, allowed_tickers, max_trade_usd, custom_instructions):
    return f"""You are a conservative portfolio rebalancing assistant. MODE: Internal Rebalance.

Sell one or more of the account's currently-owned underperforming whitelisted ETFs and use the proceeds to buy one or more of its currently-owned outperforming whitelisted ETFs. Do not introduce any new ticker — every BUY and SELL must target a ticker already held in this account.

Account state:
- Cash available: ${cash:,.2f}
- Total portfolio value: ${total_value:,.2f}
- Current whitelisted ETF positions (with weight and gain %): {json.dumps(positions, indent=2)}

Allowed tickers for this request (BUY or SELL only these): {sorted(allowed_tickers)}
{_guardrail_footer(max_trade_usd, None, custom_instructions)}"""


def build_prompt_mode2(cash, total_value, positions, allowed_tickers, max_trade_usd, market_intelligence, custom_instructions):
    candidates = market_intelligence.get("candidates", [])
    return f"""You are a conservative portfolio rebalancing assistant. MODE: External ETF Rotation.

Sell one or more of the account's currently-owned worst-performing whitelisted ETFs and use the cash to buy one or more highly-ranked ETFs NOT currently owned, chosen from the "external candidates" list below. That list was already filtered by Python to exclude anything with an expense ratio above 0.50% or more than 60% holdings overlap with the current portfolio — every candidate listed is fair game on that front.

Account state:
- Cash available: ${cash:,.2f}
- Total portfolio value: ${total_value:,.2f}
- Current whitelisted ETF positions: {json.dumps(positions, indent=2)}

S&P 500 benchmark momentum: {json.dumps(market_intelligence.get("sp500_benchmark", {}))}

External candidates (real market data, already vetted — pick your BUY target(s) from here): {json.dumps(candidates, indent=2)}

Recent ETF market headlines (qualitative color only, not a ranking): {json.dumps(market_intelligence.get("headlines", []))}

Allowed tickers for this request (SELL only from currently-held, BUY only from held+candidates): {sorted(allowed_tickers)}
{_guardrail_footer(max_trade_usd, None, custom_instructions)}
In your rationale, explicitly state the recommended BUY's relative strength vs the S&P 500 benchmark above (relative_strength_vs_sp500_3m_pct on the candidate)."""


def build_prompt_mode3(cash, total_value, positions, allowed_tickers, max_trade_usd, budget_usd, custom_instructions):
    held_tickers = {p["ticker"] for p in positions}
    unowned = sorted(t for t in allowed_tickers if t not in held_tickers)
    return f"""You are a conservative portfolio rebalancing assistant. MODE: Fresh Cash Buy.

Do NOT propose any SELL trades under any circumstances. Only propose BUY trades, deploying the account owner's stated budget into the best whitelisted ETF(s) — owned or unowned — for this account. You may split the budget across multiple ETFs if that better fits the opportunity, or concentrate it in one — whichever is actually warranted.

Account state:
- Cash available: ${cash:,.2f}
- Total portfolio value: ${total_value:,.2f}
- Current whitelisted ETF positions: {json.dumps(positions, indent=2)}
- Whitelisted ETFs not currently held: {unowned}
- Budget to deploy this run: ${budget_usd:,.2f}

Allowed tickers for this request (BUY only, from the full whitelist): {sorted(allowed_tickers)}
{_guardrail_footer(max_trade_usd, budget_usd, custom_instructions)}"""


def build_prompt_mode4(cash, total_value, positions, allowed_tickers, sub_mode, risk_tolerance, max_trade_usd, budget_usd, market_intelligence, custom_instructions):
    candidates = market_intelligence.get("candidates", [])
    band = (
        "highly conservative (bonds/defensive)" if risk_tolerance <= 25 else
        "moderate (core broad-market)" if risk_tolerance <= 55 else
        "aggressive (higher volatility, growth-tilted)" if risk_tolerance < 80 else
        "highly aggressive (leveraged/niche growth acceptable)"
    )

    if sub_mode == "4A":
        action_rules = f"""You MAY sell current holdings — especially ones that are redundant or overlapping with each other — to fund the risk-adjusted strategy. Do not propose a BUY budget line; size trades using the account's own value (${total_value:,.2f}) and cash (${cash:,.2f}).
Allowed tickers for this request (SELL only from currently-held, BUY from held+candidates): {sorted(allowed_tickers)}"""
        budget_for_footer = None
    else:  # 4B
        action_rules = f"""Do NOT propose any SELL trades. Only propose BUY trades, distributing the budget below proportionally across ETFs that match the target risk profile.
Budget to deploy this run: ${budget_usd:,.2f}
Allowed tickers for this request (BUY only, from whitelist+candidates): {sorted(allowed_tickers)}"""
        budget_for_footer = budget_usd

    return f"""You are a portfolio optimization assistant. MODE: Risk-Adjusted Optimization ({sub_mode}).

Ignore past performance and historical gains entirely. Your only objective is matching the account's risk tolerance setting: {risk_tolerance}/100 — {band}. Aggressively remove or avoid redundant/overlapping ETF exposure. If the risk tolerance is high, it is acceptable and expected to include higher-volatility, leveraged, or niche growth ETFs from the candidate list; if low, favor defensive/bond-like exposure instead.

Account state:
- Cash available: ${cash:,.2f}
- Total portfolio value: ${total_value:,.2f}
- Current whitelisted ETF positions: {json.dumps(positions, indent=2)}

S&P 500 benchmark momentum (context only — you are NOT optimizing for this in Mode 4): {json.dumps(market_intelligence.get("sp500_benchmark", {}))}

Risk-banded + external candidates (already vetted by Python for expense ratio and overlap): {json.dumps(candidates, indent=2)}

{action_rules}
{_guardrail_footer(max_trade_usd, budget_for_footer, custom_instructions)}
If you include any leveraged or niche/high-volatility ETF, explicitly flag the added risk in your rationale."""


def build_prompt_mode5(cash, total_value, positions, allowed_tickers, max_trade_usd, market_intelligence, custom_instructions):
    candidates = market_intelligence.get("candidates", [])
    held_metrics = market_intelligence.get("held_metrics", [])
    return f"""You are a forward-looking portfolio optimization assistant. MODE: Forward Opportunity Engine.

Completely ignore past gains or losses on current holdings — treat every dollar currently invested as liquid cash available to redeploy today. Rank opportunities using forward-looking data only: look-through forward P/E, look-through earnings growth, and implied 12-month return (derived from each ETF's top holdings' analyst targets — see forward_pe / earnings_growth_pct / implied_12m_return_pct below).

You may recommend PARTIAL sells: set amount_usd to whatever fraction of a position's dollar value you want to trim, or set is_full_liquidation: true to exit a position entirely. Prefer partial trims over full liquidation when only some of a position's forward outlook has weakened.

Account state:
- Cash available: ${cash:,.2f}
- Total portfolio value: ${total_value:,.2f}
- Current whitelisted ETF positions (current $ value only — ignore any embedded gain/loss): {json.dumps(positions, indent=2)}
- Forward metrics for current holdings: {json.dumps(held_metrics, indent=2)}

External candidates with forward metrics (already vetted for expense ratio and overlap): {json.dumps(candidates, indent=2)}

Allowed tickers for this request (SELL only from currently-held, BUY from held+candidates): {sorted(allowed_tickers)}
{_guardrail_footer(max_trade_usd, None, custom_instructions)}
FRICTION GUARDRAIL (also independently enforced in Python — if you violate it your trades will be discarded and treated as a HOLD anyway, so follow it): only propose a SELL-funded-BUY pair if the BUY side's implied_12m_return_pct beats the SELL side's by at least 2.5 percentage points. If nothing clears that bar, action must be HOLD."""


def build_prompt_mode6(cash, total_value, positions, allowed_tickers, max_trade_usd, budget_usd, market_intelligence, custom_instructions):
    candidates = market_intelligence.get("candidates", [])
    held_metrics = market_intelligence.get("held_metrics", [])
    macro = market_intelligence.get("macro_regime", {})
    return f"""You are a macro-aware portfolio rotation assistant. MODE: Macro Regime & Sector Rotation.

Align the portfolio's factor tilts (Growth, Small-Cap Value, Large Core, Defensive, Cash) with the CURRENT macroeconomic regime described below — not with which holdings have performed best historically. Trim or liquidate positions whose sector/factor is facing macro headwinds under the current regime, and reallocate into positions whose sector/factor is getting a macro tailwind.

Macro regime signals:
- 10-year Treasury yield: {macro.get('treasury_10y_pct')}% now vs {macro.get('treasury_10y_pct_6mo_ago')}% six months ago — trend: {macro.get('treasury_10y_trend')}. Rising yields typically pressure long-duration growth names and rate-sensitive sectors; falling yields typically help them.
- Breakeven inflation expectation: {macro.get('breakeven_inflation_pct')}% now vs {macro.get('breakeven_inflation_pct_6mo_ago')}% six months ago — trend: {macro.get('breakeven_inflation_trend')}.
- S&P 500 vs its own 200-day moving average: {macro.get('sp500_vs_200sma_pct')}% (positive = above trend / risk-on backdrop, negative = below trend / risk-off backdrop).
- 3-month sector momentum: {json.dumps(macro.get('sector_momentum_3m_pct', {}))}

Account state:
- Cash available: ${cash:,.2f}
- Total portfolio value: ${total_value:,.2f}
- Current whitelisted ETF positions: {json.dumps(positions, indent=2)}
- Technicals for current holdings: {json.dumps(held_metrics, indent=2)}

External candidates (already vetted for expense ratio and overlap): {json.dumps(candidates, indent=2)}

Allowed tickers for this request (SELL only from currently-held, BUY from held+candidates): {sorted(allowed_tickers)}
{_guardrail_footer(max_trade_usd, budget_usd, custom_instructions)}
In your rationale, explicitly connect each trade to the specific macro signal(s) driving it (e.g. "rising 10-year yields favor value/defensive over long-duration growth")."""


def build_prompt_mode7(cash, total_value, positions, allowed_tickers, max_trade_usd, budget_usd, market_intelligence, custom_instructions):
    candidates = market_intelligence.get("candidates", [])
    held_metrics = market_intelligence.get("held_metrics", [])
    return f"""You are a mean-reversion portfolio assistant. MODE: Mean-Reversion & Valuation Disparity.

Profit from short-to-medium-term overextension: take profits on overbought holdings and deploy that cash into fundamentally sound, oversold candidates. Use the pre-computed RSI-14 and Bollinger-band signals below — do not estimate these yourself.
- RSI > 70 = "overbought" -> SELL/trim candidate.
- RSI < 38 = "oversold" -> BUY candidate.
- bollinger_signal "above_upper_band" reinforces an overbought/SELL case; "below_lower_band" reinforces an oversold/BUY case.

Account state:
- Cash available: ${cash:,.2f}
- Total portfolio value: ${total_value:,.2f}
- Current whitelisted ETF positions: {json.dumps(positions, indent=2)}
- RSI / band signals for current holdings: {json.dumps(held_metrics, indent=2)}

External candidates with RSI / band signals (already vetted for expense ratio and overlap): {json.dumps(candidates, indent=2)}

Allowed tickers for this request (SELL only from currently-held, BUY from held+candidates): {sorted(allowed_tickers)}
{_guardrail_footer(max_trade_usd, budget_usd, custom_instructions)}
Only propose a SELL for a holding whose rsi_signal is "overbought" (or bollinger_signal is "above_upper_band"). Only propose a BUY for a ticker whose rsi_signal is "oversold" (or bollinger_signal is "below_lower_band"). If nothing in the account or candidate list qualifies on either side, action must be HOLD."""


def build_prompt_mode8(cash, total_value, positions, allowed_tickers, max_trade_usd, budget_usd, market_intelligence, custom_instructions):
    candidates = market_intelligence.get("candidates", [])
    held_metrics = market_intelligence.get("held_metrics", [])
    macro = market_intelligence.get("macro_regime", {})
    return f"""You are an unconstrained portfolio optimization assistant. MODE: Dynamic Tactical Rebalance.

Full-spectrum evaluation: scan current holdings against forward earnings growth, forward P/E, expense-ratio drag, and sector/factor concentration. You are NOT restricted to rebalancing only within currently-held tickers — if a candidate below (not currently owned) mathematically beats a current position by the hurdle rate, sell the weaker current position and buy the new one. You may also trim an overextended current holding to fund an underweighted one entirely within the account's existing tickers, if that is the stronger trade. Macro regime context is secondary color here, not the primary driver (that's Mode 6) — this mode's primary driver is the forward-return hurdle rate below.

Account state:
- Cash available: ${cash:,.2f}
- Total portfolio value: ${total_value:,.2f}
- Current whitelisted ETF positions: {json.dumps(positions, indent=2)}
- Forward metrics for current holdings: {json.dumps(held_metrics, indent=2)}

Macro regime context (secondary): {json.dumps(macro)}

Candidates — real market data, already vetted by Python for expense ratio (<=0.50%) and holdings overlap (<=60%) with your current portfolio. This is the full universe you're authorized to buy from for this mode, not narrowed by any risk band: {json.dumps(candidates, indent=2)}

Allowed tickers for this request (SELL only from currently-held, BUY from held+candidates): {sorted(allowed_tickers)}
{_guardrail_footer(max_trade_usd, budget_usd, custom_instructions)}
FRICTION GUARDRAIL (also independently enforced in Python — if you violate it your trades will be discarded and treated as a HOLD anyway, so follow it): only propose a SELL-funded-BUY pair if the BUY side's implied_12m_return_pct beats the SELL side's by at least 2.5 percentage points. If nothing clears that bar, action must be HOLD."""


def _call_claude(prompt: str) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

    client = Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=2000,
        tools=[PROPOSE_TRADES_TOOL],
        tool_choice={"type": "tool", "name": "propose_trades"},
        messages=[{"role": "user", "content": prompt}],
    )
    for block in resp.content:
        if block.type == "tool_use" and block.name == "propose_trades":
            return block.input
    raise RuntimeError(f"Claude did not return a propose_trades tool call: {resp.content}")


def get_rebalance_proposal(
    mode: str,
    cash: float,
    total_value: float,
    positions: list[dict],
    allowed_tickers: set[str] | list[str],
    max_trade_usd: float,
    sub_mode: str | None = None,
    budget_usd: float | None = None,
    risk_tolerance: int | None = None,
    custom_instructions: str | None = None,
    market_intelligence: dict | None = None,
) -> dict:
    """allowed_tickers must be exactly the same set the caller will later
    pass to guardrails.validate_batch — this module never computes its own
    notion of "what's allowed," it only renders whatever set it's given."""
    market_intelligence = market_intelligence or {}

    if mode == "1":
        prompt = build_prompt_mode1(cash, total_value, positions, allowed_tickers, max_trade_usd, custom_instructions)
    elif mode == "2":
        prompt = build_prompt_mode2(cash, total_value, positions, allowed_tickers, max_trade_usd, market_intelligence, custom_instructions)
    elif mode == "3":
        if budget_usd is None:
            raise ValueError("Mode 3 (Fresh Cash Buy) requires a budget_usd")
        prompt = build_prompt_mode3(cash, total_value, positions, allowed_tickers, max_trade_usd, budget_usd, custom_instructions)
    elif mode == "4":
        if sub_mode not in ("4A", "4B"):
            raise ValueError("Mode 4 requires sub_mode '4A' or '4B'")
        if sub_mode == "4B" and budget_usd is None:
            raise ValueError("Mode 4B (Fresh Cash) requires a budget_usd")
        risk_tolerance = 50 if risk_tolerance is None else risk_tolerance
        prompt = build_prompt_mode4(cash, total_value, positions, allowed_tickers, sub_mode, risk_tolerance, max_trade_usd, budget_usd, market_intelligence, custom_instructions)
    elif mode == "5":
        prompt = build_prompt_mode5(cash, total_value, positions, allowed_tickers, max_trade_usd, market_intelligence, custom_instructions)
    elif mode == "6":
        prompt = build_prompt_mode6(cash, total_value, positions, allowed_tickers, max_trade_usd, budget_usd, market_intelligence, custom_instructions)
    elif mode == "7":
        prompt = build_prompt_mode7(cash, total_value, positions, allowed_tickers, max_trade_usd, budget_usd, market_intelligence, custom_instructions)
    elif mode == "8":
        prompt = build_prompt_mode8(cash, total_value, positions, allowed_tickers, max_trade_usd, budget_usd, market_intelligence, custom_instructions)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    return _call_claude(prompt)
