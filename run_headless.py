#!/usr/bin/env python3
"""CLI entrypoint for the GitHub Actions rebalance workflow. Calls the exact
same rebalance_core.analyze() / execute_trades() functions the Flask
dashboard uses — there is no separate "headless" implementation of the
guardrail or execution logic to drift out of sync with the interactive path.

Reads its parameters from CLI args first, falling back to environment
variables (MODE, DRY_RUN, BUDGET, SUB_MODE, RISK_TOLERANCE,
CUSTOM_INSTRUCTIONS, TRADES_JSON, GITHUB_TOKEN) so it works both invoked
directly and via the workflow's `env:` block. Accepts mode as either the
internal form ("5", "4A") or the workflow_dispatch-friendly form ("mode5",
"mode4a"). With neither --mode nor --trades-json given, falls through to
run_from_queue() — the cloud-native scheduling path, reading
queue/scheduled_task.json from GitHub's Contents API (there is no local
file involved; this is what lets a scheduled run execute with the Mac that
scheduled it fully powered off).

Exit codes: 0 = ran cleanly (including a legitimate HOLD, a guardrail
rejection, or skipping because the market is closed — those are the safety
system working, not a failure). 1 = a genuine execution problem (a SELL/BUY
that failed against the broker, or an unhandled error) — this is what makes
a scheduled GitHub Actions run show up red so it actually gets noticed.
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

import guardrails
import rebalance_core
from public_client import PublicAPIError

load_dotenv()

FAILURE_STATUSES = {"FAILED_PREFLIGHT", "FAILED_ORDER", "ABORTED_SELL_FAILED", "SKIPPED_INSUFFICIENT_BUYING_POWER"}


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def normalize_mode(raw: str) -> tuple[str, str | None]:
    """"mode5" -> ("5", None). "mode4a" / "4A" -> ("4", "4A"). "3" -> ("3", None)."""
    raw = raw.strip().lower().removeprefix("mode")
    m = re.match(r"^(\d)([ab]?)$", raw)
    if not m:
        raise ValueError(f"unrecognized mode {raw!r} (expected forms like 'mode5', '5', 'mode4a', '4A')")
    digit, letter = m.group(1), m.group(2)
    if digit == "4" and letter:
        return "4", f"4{letter.upper()}"
    return digit, None


def parse_args(allow_empty: bool = False) -> argparse.Namespace:
    """allow_empty=True permits neither --mode nor --trades-json to be
    given — the expected case for a schedule-triggered Actions run, where
    it means "check the cloud queue instead" (see run_from_queue()).
    CLI/direct invocation still requires one of them."""
    parser = argparse.ArgumentParser(description="Headless rebalance runner for GitHub Actions.")
    parser.add_argument("--mode", default=os.environ.get("MODE") or None, help="e.g. mode5, mode4a, or 5 / 4A")
    parser.add_argument("--sub-mode", default=os.environ.get("SUB_MODE"), help="4A or 4B (only needed if --mode doesn't already encode it)")
    parser.add_argument("--budget", default=os.environ.get("BUDGET"), help="dollar budget; required for Mode 3 / Mode 4B")
    parser.add_argument("--risk-tolerance", default=os.environ.get("RISK_TOLERANCE") or "50")
    parser.add_argument("--custom-instructions", default=os.environ.get("CUSTOM_INSTRUCTIONS"))
    parser.add_argument(
        "--trades-json", default=os.environ.get("TRADES_JSON") or None,
        help="pre-approved batch (from rebalance_core.serialize_batch) — when given, Claude is never called; the exact trades in this JSON are re-validated by guardrails and executed as-is",
    )
    parser.add_argument(
        "--dry-run", default=os.environ.get("DRY_RUN") or "true",
        help='"true" or "false" — defaults to true; the workflow input must explicitly say "false" to place real orders',
    )
    args = parser.parse_args()
    if not allow_empty and not args.mode and not args.trades_json:
        parser.error("--mode is required unless --trades-json is given (or set MODE/TRADES_JSON env vars)")
    return args


def _log_trade_list(trades: list[dict]) -> None:
    for t in trades:
        tag = " (full liquidation)" if t.get("is_full_liquidation") else ""
        log(f"  {t['action']} ${t['amount_usd']:.2f} {t['ticker']}{tag}")


def _run_execution(batch: dict, dry_run: bool) -> tuple[bool, list[dict]]:
    """Shared execution + logging loop for both the pre-approved and
    cloud-queue paths. Returns (had_failure, final_log)."""
    had_failure = False
    final_log = []
    log("--- Execution ---")
    for evt in rebalance_core.execute_approved_batch(batch, dry_run):
        if evt["type"] == "info":
            log(evt["message"])
        elif evt["type"] == "trade_result":
            log(f"  -> {evt['action']} {evt['ticker']} ${evt.get('amount_usd')}: {evt['status']} ({evt.get('detail', '')})")
            if evt["status"] in FAILURE_STATUSES:
                had_failure = True
        elif evt["type"] == "error":
            log(f"ERROR: {evt['message']}")
            had_failure = True
        elif evt["type"] == "done":
            final_log = evt.get("log", [])
            if evt.get("warning"):
                log(f"WARNING: {evt['warning']}")
    return had_failure, final_log


def run_preapproved(trades_json: str, dry_run: bool) -> int:
    """Executes a pre-approved batch (already reviewed by a human in the
    dashboard) with zero Claude involvement — just guardrail re-validation
    and the standard sell-then-buy sequence. This is the immediate-dispatch
    path (workflow_dispatch with a trades_json input), distinct from
    run_from_queue()'s scheduled-cron path."""
    try:
        batch = json.loads(trades_json)
    except json.JSONDecodeError as e:
        log(f"ERROR: --trades-json is not valid JSON: {e}")
        return 1

    trades = batch.get("trades", [])
    log(f"Pre-approved batch: {len(trades)} trade(s). Skipping Claude entirely.")
    _log_trade_list(trades)
    if not trades:
        log("No trades in the pre-approved payload. Done.")
        return 0

    had_failure, _ = _run_execution(batch, dry_run)
    log("Done." if not had_failure else "Done, with failures — see log above.")
    return 1 if had_failure else 0


def run_from_queue() -> int:
    """Cloud-native scheduling path: reads queue/scheduled_task.json from
    GitHub's Contents API — not a local file, since this may be running
    with the machine that scheduled it fully powered off. This is the
    ONLY scheduling mechanism in the app (see github_service.py's module
    docstring for why an earlier local-timer version was removed).

    Claims the task atomically before executing: PUTs the file back with
    status "in_progress" using the exact sha just read. GitHub's Contents
    API rejects a PUT whose sha doesn't match the file's current version
    (409), so if this ever runs concurrently with another claim attempt
    (overlapping cron polls, or a manual Run Now landing at the same
    moment), only the first to land actually claims it — the loser backs
    off cleanly instead of double-executing the same trades."""
    import github_service

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        log("ERROR: GITHUB_TOKEN is not set — can't read the cloud task queue.")
        return 1

    task, sha = github_service.get_queue_file_raw(token=token)
    if not task:
        log("No task in the cloud queue (queue/scheduled_task.json). Nothing to do.")
        return 0
    if task.get("status") != "pending":
        log(f"Queue task status is {task.get('status')!r}, not 'pending'. Nothing to do.")
        return 0

    try:
        scheduled_for = datetime.fromisoformat(task["scheduled_for"].replace("Z", "+00:00"))
    except (KeyError, ValueError) as e:
        log(f"ERROR: bad scheduled_for in queue file: {e}")
        return 1
    if datetime.now(timezone.utc) < scheduled_for:
        log(f"Task exists but isn't due yet (scheduled_for={task['scheduled_for']}). Nothing to do.")
        return 0

    log(f"Claiming cloud-queued task (scheduled_for={task['scheduled_for']})...")
    task["status"] = "in_progress"
    try:
        github_service.commit_scheduled_task(task, token=token, expected_sha=sha)
    except github_service.GitHubServiceError as e:
        log(f"Could not claim the task (most likely another run already claimed it): {e}")
        return 0

    dry_run = bool(task.get("dry_run", True))
    trades = task.get("trades", [])
    log(f"Claimed: {len(trades)} trade(s), dry_run={dry_run}. Skipping Claude entirely.")
    _log_trade_list(trades)

    had_failure = False
    if trades:
        had_failure, _ = _run_execution(task, dry_run)
    else:
        log("No trades in the queued task.")

    task["status"] = "failed" if had_failure else "completed"
    task["completed_at"] = datetime.now(timezone.utc).isoformat()
    try:
        github_service.commit_scheduled_task(task, token=token)
        log(f"Queue task marked {task['status']}.")
    except github_service.GitHubServiceError as e:
        log(f"WARNING: could not update queue file after execution: {e}")

    log("Done." if not had_failure else "Done, with failures — see log above.")
    return 1 if had_failure else 0


def main() -> int:
    args = parse_args(allow_empty=True)
    dry_run = str(args.dry_run).strip().lower() != "false"

    if args.trades_json:
        return run_preapproved(args.trades_json, dry_run)

    if not args.mode:
        return run_from_queue()

    mode, mode_encoded_sub = normalize_mode(args.mode)
    sub_mode = mode_encoded_sub or (args.sub_mode.strip().upper() if args.sub_mode else None)
    budget_usd = float(args.budget) if args.budget not in (None, "", "null") else None
    risk_tolerance = int(args.risk_tolerance)

    log(f"Mode {mode}{'/' + sub_mode if sub_mode else ''} | dry_run={dry_run} | budget={budget_usd} | risk_tolerance={risk_tolerance}")

    if not dry_run and not guardrails.is_market_hours():
        log("Market is closed (Mon-Fri 9:30am-4:00pm America/New_York) and dry_run=false — skipping this run entirely. Nothing was analyzed or executed.")
        return 0

    try:
        client = rebalance_core.get_client()
        body = {
            "mode": mode,
            "sub_mode": sub_mode,
            "budget_usd": budget_usd,
            "risk_tolerance": risk_tolerance,
            "custom_instructions": args.custom_instructions,
        }
        result = rebalance_core.analyze(body, client=client)
    except (rebalance_core.BadRequest, PublicAPIError) as e:
        log(f"ERROR during analysis: {e}")
        return 1

    log("--- Rationale ---")
    log(result["rationale"] or "(none)")
    log("--- Proposed trades ---")
    for t in result["trades"]:
        tag = " (full liquidation)" if t.get("is_full_liquidation") else ""
        log(f"  {t['action']} ${t['amount_usd']:.2f} {t['ticker']}{tag} — {t.get('rationale', '')}")
    if result["rejected"]:
        log("--- Rejected by guardrails ---")
        for r in result["rejected"]:
            log(f"  {r['action']} {r['ticker']} ${r.get('amount_usd')} — {r.get('rejection_reason')}")

    if not result["trades"]:
        log("No trades to execute (HOLD). Done.")
        _write_step_summary(mode, sub_mode, dry_run, result, [])
        return 0

    had_failure, exec_log = _run_execution(
        {"trades": result["trades"], "allowed_tickers": result["allowed_tickers"], "max_trade_usd": result["max_trade_usd"], "budget_usd": result["budget_usd"]},
        dry_run,
    )

    _write_step_summary(mode, sub_mode, dry_run, result, exec_log)
    log("Done." if not had_failure else "Done, with failures — see log above.")
    return 1 if had_failure else 0


def _write_step_summary(mode, sub_mode, dry_run, result, exec_log) -> None:
    """Writes a Markdown summary to $GITHUB_STEP_SUMMARY when running
    inside GitHub Actions, so the run shows a readable summary in the
    Actions UI instead of requiring someone to read the raw log."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    lines = [
        f"# Rebalance run — Mode {mode}{'/' + sub_mode if sub_mode else ''}",
        "",
        f"**Dry run:** {dry_run}  ",
        f"**Total portfolio value:** ${result['total_value']:,.2f}  ",
        f"**Cash:** ${result['cash']:,.2f}",
        "",
        "## Rationale",
        result["rationale"] or "(none)",
        "",
        "## Trades",
        "",
        "| Action | Ticker | Amount | Status |",
        "|---|---|---|---|",
    ]
    status_by_ticker_action = {(e["action"], e["ticker"], e.get("amount_usd")): e.get("status", "?") for e in exec_log}
    if not result["trades"]:
        lines.append("| — | — | — | HOLD — no trades proposed |")
    for t in result["trades"]:
        status = status_by_ticker_action.get((t["action"], t["ticker"], t["amount_usd"]), "not executed")
        lines.append(f"| {t['action']} | {t['ticker']} | ${t['amount_usd']:.2f} | {status} |")
    try:
        with open(summary_path, "a") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        pass  # best-effort only; never fail the run over the summary file


if __name__ == "__main__":
    sys.exit(main())
