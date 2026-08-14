"""Local rebalancing dashboard. Runs on http://localhost:5000 by default.

Auth: if APP_PASSWORD is set in .env, every route requires either a logged
-in session or a `?key=<password>` query param (so a bookmarked URL works
from a phone over Tailscale without re-typing a password each time). If
APP_PASSWORD is unset, the app behaves exactly as before — no auth, assumed
localhost-only. This is basic protection suitable for a private Tailscale
network, not a substitute for real authentication if ever exposed to the
open internet — there's no rate limiting or lockout on failed attempts.
"""
import hmac
import json
import os
import time
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, redirect, render_template, request, session, stream_with_context, url_for

import github_service
import guardrails
import rebalance_core
from public_client import PublicAPIError, parse_portfolio

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-insecure-key")

DRY_RUN = os.environ.get("DRY_RUN", "true").strip().lower() != "false"
APP_PASSWORD = os.environ.get("APP_PASSWORD") or None

# In-memory store for proposed trade batches, keyed by batch_id. Captures
# everything needed to re-validate identically at execute time — the
# allowed-ticker set and dollar caps are NOT recomputed at execute time,
# so there's no window where a re-run of market intelligence could
# silently approve a different set of tickers than what was shown to the
# user for approval.
_BATCHES: dict[str, dict] = {}


# -- auth gate ----------------------------------------------------------

def _is_authenticated() -> bool:
    if not APP_PASSWORD:
        return True
    if session.get("authenticated"):
        return True
    key = request.args.get("key", "")
    if key and hmac.compare_digest(key, APP_PASSWORD):
        session["authenticated"] = True
        session.permanent = True
        return True
    return False


@app.before_request
def require_auth():
    if not APP_PASSWORD or request.endpoint in ("login", "static"):
        return None
    if _is_authenticated():
        return None
    if request.path.startswith("/api/"):
        return jsonify({"error": "unauthorized"}), 401
    return redirect(url_for("login", next=request.path))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        password = request.form.get("password", "")
        if APP_PASSWORD and hmac.compare_digest(password, APP_PASSWORD):
            session["authenticated"] = True
            session.permanent = True
            return redirect(request.form.get("next") or url_for("index"))
        error = "Incorrect password"
    return render_template("login.html", error=error, next=request.args.get("next", ""))


@app.route("/logout")
def logout():
    session.pop("authenticated", None)
    return redirect(url_for("login"))


# -- dashboard routes -----------------------------------------------------

@app.route("/")
def index():
    return render_template(
        "index.html",
        dry_run=DRY_RUN,
        max_trade_usd=guardrails.get_max_trade_usd(),
        whitelist=sorted(guardrails.get_whitelist()),
        github_configured=github_service.is_configured(),
        github_actions_url=github_service.actions_url(),
    )


@app.route("/api/state")
def api_state():
    """Read-only snapshot: cash, total value, whitelisted positions."""
    try:
        client = rebalance_core.get_client()
        normalized = parse_portfolio(client.get_portfolio())
        whitelist = guardrails.get_whitelist()
        positions = rebalance_core.filter_and_weight(normalized["positions"], whitelist, normalized["total_value"])

        symbols = [p["ticker"] for p in positions]
        if symbols:
            quotes = client.get_quotes(symbols)
            for p in positions:
                q = quotes.get(p["ticker"])
                if q:
                    p["last_price"] = q.last

        return jsonify({
            "cash": normalized["cash"],
            "total_value": normalized["total_value"],
            "positions": positions,
            "market_hours_open": guardrails.is_market_hours(),
            "dry_run": DRY_RUN,
        })
    except PublicAPIError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Fetch fresh portfolio state, build the mode-appropriate candidate
    universe, ask Claude for a proposal, and hard-filter the response
    through guardrails before showing it to the user. Nothing is executed
    here. Delegates to rebalance_core.analyze() — the same function
    run_headless.py calls for the GitHub Actions path."""
    body = request.get_json(silent=True) or {}
    try:
        result = rebalance_core.analyze(body)

        batch_id = str(uuid.uuid4())
        _BATCHES[batch_id] = {
            "trades": result["trades"],
            "allowed_tickers": result["allowed_tickers"],
            "max_trade_usd": result["max_trade_usd"],
            "budget_usd": result["budget_usd"],
            "mode": result["mode"],
            "sub_mode": result["sub_mode"],
            "created_at": time.time(),
        }

        mi = result["market_intelligence"]
        return jsonify({
            "batch_id": batch_id,
            "rationale": result["rationale"],
            "trades": result["trades"],
            "rejected": result["rejected"],
            "cash": result["cash"],
            "total_value": result["total_value"],
            "market_intelligence": {
                "sources_used": mi.get("sources_used", {}),
                "sp500_benchmark": mi.get("sp500_benchmark", {}),
                "headlines": mi.get("headlines", []),
                "rejected_candidates": mi.get("rejected_candidates", []),
                "macro_regime": mi.get("macro_regime"),
                "held_metrics": mi.get("held_metrics", []),
            } if mi else None,
        })
    except rebalance_core.BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except PublicAPIError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/execute", methods=["POST"])
def api_execute():
    """Streams rebalance_core.execute_trades()'s events as newline
    -delimited JSON. See that function's docstring for the 2-phase
    sell-then-buy sequence (with live full-liquidation resolution and
    dynamic buying-power capping)."""
    body = request.get_json(silent=True) or {}
    batch_id = body.get("batch_id")
    batch = _BATCHES.get(batch_id)
    if not batch:
        return jsonify({"error": "unknown or expired batch_id; run Analyze again"}), 404

    if not DRY_RUN and not guardrails.is_market_hours():
        return jsonify({"error": "market is closed (Mon-Fri 9:30am-4:00pm America/New_York); refusing to execute"}), 409

    def stream():
        try:
            client = rebalance_core.get_client()
            for evt in rebalance_core.execute_trades(
                client, batch["trades"], batch["allowed_tickers"], batch["max_trade_usd"], batch["budget_usd"], DRY_RUN,
            ):
                yield json.dumps(evt) + "\n"
                if evt.get("type") == "done":
                    _BATCHES.pop(batch_id, None)
        except PublicAPIError as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return Response(stream_with_context(stream()), mimetype="application/x-ndjson")


# -- GitHub Actions cloud automation --------------------------------------

@app.route("/api/github/status")
def api_github_status():
    return jsonify({"configured": github_service.is_configured()})


def _trades_json_for(body: dict) -> str | None:
    """If the request references a batch_id from a prior /api/analyze call,
    serializes it for the GitHub dispatch payload so run_headless.py skips
    Claude and runs the exact reviewed trades instead."""
    batch_id = body.get("batch_id")
    if not batch_id:
        return None
    batch = _BATCHES.get(batch_id)
    if not batch:
        raise rebalance_core.BadRequest("unknown or expired batch_id; run Analyze again")
    return json.dumps(rebalance_core.serialize_batch(batch))


@app.route("/api/github/trigger", methods=["POST"])
def api_github_trigger():
    body = request.get_json(silent=True) or {}
    try:
        result = github_service.trigger_github_action(
            mode=body.get("mode", "1"),
            dry_run=bool(body.get("dry_run", True)),
            budget=body.get("budget"),
            trades_json=_trades_json_for(body),
        )
        return jsonify(result)
    except rebalance_core.BadRequest as e:
        return jsonify({"error": str(e)}), 404
    except github_service.GitHubServiceError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -- Cloud-native scheduling (the ONLY scheduling mechanism in this app — --
# -- see github_service.py's module docstring for why a local-timer       --
# -- version was tried and removed). Commits the approved batch directly  --
# -- into the repo via GitHub's Contents API; .github/workflows/          --
# -- rebalance.yml's `schedule:` cron reads it entirely inside GitHub's   --
# -- cloud — once committed, this machine can be fully shut down. --------

@app.route("/api/cloud-schedule", methods=["GET"])
def api_cloud_schedule_get():
    """The dashboard's "Next Scheduled Task" panel polls this."""
    try:
        task = github_service.get_scheduled_task()
        return jsonify({"task": task, "actions_url": github_service.actions_url()})
    except github_service.GitHubServiceError as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/cloud-schedule", methods=["POST"])
def api_cloud_schedule_post():
    body = request.get_json(silent=True) or {}
    batch_id = body.get("batch_id")
    batch = _BATCHES.get(batch_id)
    if not batch:
        return jsonify({"error": "unknown or expired batch_id; run Analyze again"}), 404
    try:
        existing = github_service.get_scheduled_task()
        if existing and existing.get("status") in ("pending", "in_progress"):
            return jsonify({"error": f"a cloud task is already {existing['status']} — cancel it first"}), 400

        run_date = github_service.resolve_run_date(body.get("custom_datetime") or None)
        trades = batch["trades"]
        payload = {
            "status": "pending",
            "scheduled_for": run_date.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "dry_run": bool(body.get("dry_run", DRY_RUN)),
            "total_sell_usd": round(sum(t["amount_usd"] for t in trades if t["action"] == "SELL"), 2),
            "total_buy_usd": round(sum(t["amount_usd"] for t in trades if t["action"] == "BUY"), 2),
            "trades": trades,
            # Extends the requested schema with the fields guardrail
            # re-validation actually needs at execution time (the ticker
            # whitelist/candidate set and dollar caps in effect when this
            # was approved) — without these, execute_approved_batch()
            # would have nothing safe to re-validate against.
            "allowed_tickers": sorted(batch["allowed_tickers"]),
            "max_trade_usd": batch["max_trade_usd"],
            "budget_usd": batch["budget_usd"],
        }
        github_service.commit_scheduled_task(payload)
        return jsonify({"ok": True, "task": payload})
    except github_service.GitHubServiceError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cloud-schedule", methods=["DELETE"])
def api_cloud_schedule_delete():
    try:
        cancelled = github_service.cancel_scheduled_task()
        if not cancelled:
            return jsonify({"error": "no cloud task is scheduled"}), 404
        return jsonify({"ok": True})
    except github_service.GitHubServiceError as e:
        return jsonify({"error": str(e)}), 502


if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
