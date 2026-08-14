"""GitHub integration for the rebalance dashboard: immediate workflow
dispatch, and the cloud-native task queue that's the sole scheduling
mechanism in this app.

Two things live here, and they behave differently:
  - trigger_github_action() fires the workflow_dispatch event immediately
    over the network — it works regardless of whether this local Flask
    process keeps running afterward (the run itself is already dispatched
    to GitHub the moment this call returns).
  - commit_scheduled_task() / get_scheduled_task() / cancel_scheduled_task()
    are the Mac-sleep-proof scheduling mechanism: they write/read/delete
    queue/scheduled_task.json directly IN THE REPO via the GitHub Contents
    API. .github/workflows/rebalance.yml has a `schedule:` cron trigger
    that reads this file entirely inside GitHub's cloud — once a task is
    committed, this machine can be fully shut down. There is no local
    timer, no local file, and no other scheduling path in this app —
    intentionally: an in-process scheduler (APScheduler) was tried in an
    earlier version and removed, because it silently required this Flask
    process to still be running at the scheduled time, defeating the
    entire point of "schedule this, then shut the Mac down."

trigger_github_action() and the schedule payload can both carry a
pre-approved trades_json/serialize_batch() payload (a batch already
reviewed by a human in the UI) instead of a bare mode/budget/risk
selection — when present, run_headless.py skips Claude entirely and just
executes what was already approved.

GH_PAT needs the "repo" scope for a private repository (or "public_repo" +
"workflow" for a public one) to dispatch workflow runs and read/write repo
contents via the API. Named GH_PAT rather than GITHUB_PAT because GitHub
reserves the GITHUB_ prefix for its own repository secrets — trying to
create a repo secret literally named GITHUB_PAT fails outright ("Secret
names must not start with GITHUB_"). This only applies to the local .env
name; inside an Actions run itself, GitHub's own auto-provided
secrets.GITHUB_TOKEN is used instead (see run_headless.py) — this module's
GH_PAT is for the local dashboard's own calls, not something read inside
the workflow.
"""
import base64
import json
import os
from datetime import datetime, timedelta

import requests

from guardrails import MARKET_TZ

API_BASE = "https://api.github.com"
QUEUE_PATH = "queue/scheduled_task.json"


class GitHubServiceError(RuntimeError):
    pass


def _config():
    return {
        "owner": os.environ.get("GITHUB_OWNER", "Botsie696"),
        "repo": os.environ.get("GITHUB_REPO", "stocker"),
        "workflow_file": os.environ.get("GITHUB_WORKFLOW_FILE", "rebalance.yml"),
        "ref": os.environ.get("GITHUB_REF", "main"),
        "pat": os.environ.get("GH_PAT"),
    }


def actions_url() -> str:
    cfg = _config()
    return f"https://github.com/{cfg['owner']}/{cfg['repo']}/actions"


def is_configured() -> bool:
    cfg = _config()
    return bool(cfg["pat"] and cfg["owner"] and cfg["repo"])


def next_market_open_plus_5(now: datetime | None = None) -> datetime:
    """Next weekday at 9:35am America/New_York (5 minutes after the
    9:30am open). Does NOT account for market holidays — same documented
    gap as guardrails.is_market_hours(); a holiday-aware run will just
    fire against a closed market and get rejected before any order is
    placed, so it fails safe, not silently."""
    now = now.astimezone(MARKET_TZ) if now else datetime.now(MARKET_TZ)
    target = now.replace(hour=9, minute=35, second=0, microsecond=0)
    if now >= target:
        target += timedelta(days=1)
    while target.weekday() >= 5:  # Sat/Sun
        target += timedelta(days=1)
    return target


def resolve_run_date(custom_datetime: str | None) -> datetime:
    """custom_datetime, if given, is an ISO-8601 string (as produced by an
    HTML datetime-local input) interpreted in America/New_York if it has
    no timezone offset — every scheduling control in this UI is labeled
    as ET regardless of the viewer's own device timezone."""
    if custom_datetime:
        run_date = datetime.fromisoformat(custom_datetime)
        if run_date.tzinfo is None:
            run_date = run_date.replace(tzinfo=MARKET_TZ)
        if run_date <= datetime.now(MARKET_TZ):
            raise GitHubServiceError("custom_datetime must be in the future")
        return run_date
    return next_market_open_plus_5()


# -- immediate dispatch -------------------------------------------------

def trigger_github_action(mode: str, dry_run: bool, budget: float | None, trades_json: str | None = None) -> dict:
    """Fires the rebalance.yml workflow_dispatch event immediately. When
    trades_json is given (a JSON-encoded pre-approved batch from
    rebalance_core.serialize_batch()), run_headless.py uses it directly and
    skips calling Claude — mode/dry_run/budget still go through as a
    fallback/log record even though the workflow ignores mode in that case.
    Raises GitHubServiceError on any non-2xx response (bad PAT, workflow
    file not found, repo not found, etc.)."""
    cfg = _config()
    if not cfg["pat"]:
        raise GitHubServiceError("GH_PAT is not set (see .env.example)")

    url = f"{API_BASE}/repos/{cfg['owner']}/{cfg['repo']}/actions/workflows/{cfg['workflow_file']}/dispatches"
    payload = {
        "ref": cfg["ref"],
        "inputs": {
            "mode": str(mode),
            "dry_run": "true" if dry_run else "false",
            "budget": str(budget) if budget is not None else "",
            "trades_json": trades_json or "",
        },
    }
    resp = requests.post(
        url,
        json=payload,
        headers={
            "Authorization": f"Bearer {cfg['pat']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=15,
    )
    if resp.status_code != 204:
        raise GitHubServiceError(f"GitHub dispatch failed: HTTP {resp.status_code} — {resp.text}")
    return {
        "ok": True,
        "dispatched_at": datetime.now(MARKET_TZ).isoformat(),
        "mode": mode,
        "dry_run": dry_run,
        "budget": budget,
        "pre_approved": trades_json is not None,
        "repo": f"{cfg['owner']}/{cfg['repo']}",
    }


# -- cloud-native task queue (GitHub Contents API) — the only scheduling ---
# -- mechanism in this app; see module docstring for why -------------------

def _gh_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _resolve_token(token: str | None) -> str:
    tok = token or _config()["pat"]
    if not tok:
        raise GitHubServiceError("no GitHub token available (GH_PAT is not set, and no token was passed explicitly)")
    return tok


def get_queue_file_raw(token: str | None = None) -> tuple[dict | None, str | None]:
    """Returns (parsed_task_or_None, sha_or_None). sha is needed for any
    subsequent PUT/DELETE — GitHub's Contents API requires it and uses it
    for optimistic concurrency (a stale sha on PUT/DELETE gets a 409,
    which run_headless.py relies on to prevent double-execution when
    claiming a due task)."""
    cfg = _config()
    tok = _resolve_token(token)
    url = f"{API_BASE}/repos/{cfg['owner']}/{cfg['repo']}/contents/{QUEUE_PATH}"
    resp = requests.get(url, headers=_gh_headers(tok), params={"ref": cfg["ref"]}, timeout=15)
    if resp.status_code == 404:
        return None, None
    if resp.status_code != 200:
        raise GitHubServiceError(f"failed to read {QUEUE_PATH}: HTTP {resp.status_code} — {resp.text}")
    data = resp.json()
    task = json.loads(base64.b64decode(data["content"]).decode())
    return task, data["sha"]


def get_scheduled_task(token: str | None = None) -> dict | None:
    """Read-only view of the queued task, for the dashboard's "Next
    Scheduled Task" panel. None if no task is queued."""
    task, _sha = get_queue_file_raw(token)
    return task


def commit_scheduled_task(payload: dict, token: str | None = None, expected_sha: str | None = ...) -> dict:
    """Creates or overwrites queue/scheduled_task.json with payload. Used
    both to schedule a new task and to update an existing one's status
    (pending -> in_progress -> completed/failed).

    expected_sha, when explicitly passed (including None for "must not
    already exist"), is used as-is instead of being looked up fresh — this
    is what lets run_headless.py's claim step be a true atomic
    compare-and-swap: it reads the file once, then PUTs with the sha it
    just saw, so a second concurrent claim attempt (different sha by then)
    gets a 409 instead of silently overwriting the first claim. The
    default (the ... sentinel) looks up the current sha fresh, which is
    fine for the dashboard's own schedule/cancel/status-update calls where
    only one writer is realistically ever in flight."""
    cfg = _config()
    tok = _resolve_token(token)
    if expected_sha is ...:
        _existing, sha = get_queue_file_raw(tok)
    else:
        sha = expected_sha
    url = f"{API_BASE}/repos/{cfg['owner']}/{cfg['repo']}/contents/{QUEUE_PATH}"
    body = {
        "message": f"chore: {payload.get('status', 'update')} scheduled rebalance task",
        "content": base64.b64encode(json.dumps(payload, indent=2).encode()).decode(),
        "branch": cfg["ref"],
    }
    if sha:
        body["sha"] = sha
    resp = requests.put(url, headers=_gh_headers(tok), json=body, timeout=15)
    if resp.status_code == 409:
        raise GitHubServiceError("queue file was modified concurrently (sha conflict) — someone/something else updated it first")
    if resp.status_code not in (200, 201):
        raise GitHubServiceError(f"failed to write {QUEUE_PATH}: HTTP {resp.status_code} — {resp.text}")
    return resp.json()


def cancel_scheduled_task(token: str | None = None) -> bool:
    """Deletes queue/scheduled_task.json. Returns False if nothing was
    queued. Only meaningfully cancels a task still in "pending" status —
    once a task is "in_progress" the Actions run has already claimed it;
    deleting the file at that point doesn't stop the run (use the GitHub
    Actions link in the dashboard to intervene there instead)."""
    cfg = _config()
    tok = _resolve_token(token)
    _existing, sha = get_queue_file_raw(tok)
    if not sha:
        return False
    url = f"{API_BASE}/repos/{cfg['owner']}/{cfg['repo']}/contents/{QUEUE_PATH}"
    resp = requests.delete(
        url, headers=_gh_headers(tok),
        json={"message": "chore: cancel scheduled rebalance task", "sha": sha, "branch": cfg["ref"]},
        timeout=15,
    )
    if resp.status_code not in (200, 204):
        raise GitHubServiceError(f"failed to delete {QUEUE_PATH}: HTTP {resp.status_code} — {resp.text}")
    return True
