"""Thin wrapper around the Public.com brokerage API.

Auth is two-step: a long-lived "secret" (from Public.com > Settings >
Developer/API) is exchanged for a short-lived JWT access token, which is
then sent as `Authorization: Bearer <token>` on every trading call. This
client caches the access token and re-exchanges it a little before it
expires.

Endpoint paths below were pulled from the live docs at public.com/api/docs
(accounts, portfolio v2, quotes, order placement, preflight, order status)
as of 2026-08. Public.com does not publish a request/response schema for
every field (e.g. whether preflight accepts a dollar `amount` the same way
order placement does) — verify against your own account in DRY_RUN mode
before trusting this against real orders, and re-check the docs if any
call below starts returning 4xx.
"""
import json
import os
import time
import uuid
from dataclasses import dataclass

import requests

BASE_URL = "https://api.public.com"
TOKEN_TTL_SECONDS = 15 * 60  # matches the API's default validityInMinutes
TOKEN_REFRESH_MARGIN = 60  # re-auth 60s before expiry, not exactly at it


def _debug_enabled() -> bool:
    return os.environ.get("DEBUG_API", "true").strip().lower() == "true"


def _debug_dump(label: str, data) -> None:
    """Prints the raw API payload to the terminal. On by default
    (DEBUG_API=false to silence) because the doc-published example schemas
    have been observed to disagree with what accounts actually return —
    this is the fastest way to see the real field names for a given
    account and adjust the extractors below."""
    if not _debug_enabled():
        return
    print(f"\n----- DEBUG: {label} -----")
    print(json.dumps(data, indent=2, default=str))
    print(f"----- end {label} -----\n")


class PublicAPIError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


@dataclass
class Quote:
    symbol: str
    last: float
    bid: float | None = None
    ask: float | None = None


def _first_float(d: dict, keys: list[str]) -> float | None:
    for key in keys:
        if key in d and d[key] is not None:
            try:
                return float(d[key])
            except (TypeError, ValueError):
                continue
    return None


def _extract_cash(portfolio: dict) -> float:
    val = _first_float(portfolio, ["cash", "cashBalance", "availableCash", "uninvestedCash"])
    if val is not None:
        return val
    buying_power = portfolio.get("buyingPower")
    if isinstance(buying_power, dict):
        val = _first_float(buying_power, ["cashOnlyBuyingPower", "buyingPower"])
        if val is not None:
            return val
    return 0.0


def _extract_symbol(position: dict) -> str | None:
    instrument = position.get("instrument")
    if isinstance(instrument, dict) and instrument.get("symbol"):
        return str(instrument["symbol"]).upper()
    for key in ("symbol", "ticker"):
        if position.get(key):
            return str(position[key]).upper()
    return None


def _extract_quantity(position: dict) -> float:
    return _first_float(position, ["quantity", "qty", "shares"]) or 0.0


def _extract_position_value(position: dict) -> float:
    return _first_float(position, ["currentValue", "marketValue", "value", "positionValue"]) or 0.0


def _extract_last_price(position: dict) -> float:
    last_price = position.get("lastPrice")
    if isinstance(last_price, dict):
        val = _first_float(last_price, ["lastPrice", "price"])
        if val is not None:
            return val
    return _first_float(position, ["lastPrice", "price", "currentPrice"]) or 0.0


def _extract_gain_pct(position: dict) -> float | None:
    cost_basis = position.get("costBasis")
    if isinstance(cost_basis, dict):
        val = _first_float(cost_basis, ["gainPercentage", "gainPercent"])
        if val is not None:
            return val
    return _first_float(position, ["gainPercentage", "unrealizedGainPercent"])


def parse_portfolio(raw: dict) -> dict:
    """Normalizes a raw /portfolio/v2 response into
    {cash, total_value, positions: [{ticker, quantity, market_value,
    last_price, gain_pct}]}.

    Every field is looked up under several plausible key names rather than
    one, because live responses have been observed to disagree with the
    published doc examples. total_value is *computed* as
    cash + sum(all position values) rather than trusted from a single
    top-level field (e.g. "totalAccountValue") — that keeps it correct even
    when that field is named differently than expected, and it naturally
    includes non-whitelisted holdings (individual stocks, etc.) that the
    caller may filter out of the display/trading view later.

    If this still doesn't line up with your account, set DEBUG_API=true
    (default) and compare the printed raw response against the key names
    tried here."""
    cash = _extract_cash(raw)
    positions = []
    for pos in raw.get("positions", []) or []:
        symbol = _extract_symbol(pos)
        if not symbol:
            continue
        positions.append({
            "ticker": symbol,
            "quantity": _extract_quantity(pos),
            "market_value": _extract_position_value(pos),
            "last_price": _extract_last_price(pos),
            "gain_pct": _extract_gain_pct(pos),
        })
    total_value = cash + sum(p["market_value"] for p in positions)
    return {"cash": cash, "total_value": total_value, "positions": positions}


class PublicClient:
    def __init__(self, secret: str, account_id: str | None = None, timeout: int = 15):
        if not secret:
            raise ValueError("PUBLIC_API_SECRET is required")
        self._secret = secret
        self._account_id = account_id
        self._timeout = timeout
        self._access_token: str | None = None
        self._token_expires_at: float = 0.0
        self._session = requests.Session()

    # -- auth -----------------------------------------------------------

    def _get_access_token(self) -> str:
        if self._access_token and time.time() < self._token_expires_at - TOKEN_REFRESH_MARGIN:
            return self._access_token

        resp = self._session.post(
            f"{BASE_URL}/userapiauthservice/personal/access-tokens",
            json={"secret": self._secret, "validityInMinutes": TOKEN_TTL_SECONDS // 60},
            timeout=self._timeout,
        )
        self._raise_for_status(resp, "authenticating with Public.com")
        data = resp.json()
        token = data.get("accessToken")
        if not token:
            raise PublicAPIError(f"no accessToken in auth response: {data}")
        self._access_token = token
        self._token_expires_at = time.time() + TOKEN_TTL_SECONDS
        return token

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._get_access_token()}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _raise_for_status(resp: requests.Response, action: str):
        if resp.ok:
            return
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        raise PublicAPIError(
            f"Public.com API error while {action}: HTTP {resp.status_code} — {body}",
            status_code=resp.status_code,
            body=body,
        )

    # -- account / portfolio ---------------------------------------------

    def get_accounts(self) -> list[dict]:
        resp = self._session.get(
            f"{BASE_URL}/userapigateway/trading/account",
            headers=self._headers(),
            timeout=self._timeout,
        )
        self._raise_for_status(resp, "fetching accounts")
        data = resp.json()
        _debug_dump("GET /accounts response", data)
        return data.get("accounts", [])

    def resolve_account_id(self) -> str:
        if self._account_id:
            return self._account_id
        accounts = self.get_accounts()
        if not accounts:
            raise PublicAPIError("no accounts returned for this token")

        if len(accounts) == 1:
            self._account_id = accounts[0]["accountId"]
            return self._account_id

        # Multiple accounts under one token (e.g. a HIGH_YIELD cash
        # account alongside a BROKERAGE trading account) — picking
        # accounts[0] blindly can silently land on an empty, non-tradable
        # account. Prefer a BROKERAGE account that actually allows
        # trading; fall back to any tradable account, then to the first.
        tradable = [a for a in accounts if a.get("tradePermissions") == "BUY_AND_SELL"]
        brokerage_tradable = [a for a in tradable if a.get("accountType") == "BROKERAGE"]
        chosen = (brokerage_tradable or tradable or accounts)[0]
        _debug_dump(
            "account auto-selection (multiple accounts found — set PUBLIC_ACCOUNT_ID to override)",
            {"accounts": accounts, "chosen": chosen.get("accountId")},
        )
        self._account_id = chosen["accountId"]
        return self._account_id

    def get_portfolio(self) -> dict:
        account_id = self.resolve_account_id()
        resp = self._session.get(
            f"{BASE_URL}/userapigateway/trading/{account_id}/portfolio/v2",
            headers=self._headers(),
            timeout=self._timeout,
        )
        self._raise_for_status(resp, "fetching portfolio")
        data = resp.json()
        _debug_dump(f"GET /portfolio/v2 response (account {account_id})", data)
        return data

    # -- market data ------------------------------------------------------

    def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        account_id = self.resolve_account_id()
        resp = self._session.post(
            f"{BASE_URL}/userapigateway/marketdata/{account_id}/quotes",
            headers=self._headers(),
            json={"instruments": [{"symbol": s, "type": "EQUITY"} for s in symbols]},
            timeout=self._timeout,
        )
        self._raise_for_status(resp, "fetching quotes")
        out = {}
        for q in resp.json().get("quotes", []):
            sym = q["instrument"]["symbol"]
            out[sym] = Quote(
                symbol=sym,
                last=float(q["last"]),
                bid=float(q["bid"]) if q.get("bid") is not None else None,
                ask=float(q["ask"]) if q.get("ask") is not None else None,
            )
        return out

    # -- orders -------------------------------------------------------------

    def preflight_order(self, symbol: str, side: str, amount_usd: float) -> dict:
        account_id = self.resolve_account_id()
        resp = self._session.post(
            f"{BASE_URL}/userapigateway/trading/{account_id}/preflight/single-leg",
            headers=self._headers(),
            json={
                "instrument": {"symbol": symbol, "type": "EQUITY"},
                "orderSide": side,
                "orderType": "MARKET",
                "amount": str(amount_usd),
                "expiration": {"timeInForce": "DAY"},
                "validateOrder": True,
            },
            timeout=self._timeout,
        )
        self._raise_for_status(resp, f"preflighting {side} {symbol}")
        return resp.json()

    def place_order(self, symbol: str, side: str, amount_usd: float) -> str:
        """Places a market order sized in dollars. Returns the orderId.
        Order placement is asynchronous per Public's docs — this call only
        confirms submission, not execution; poll get_order() for status."""
        account_id = self.resolve_account_id()
        order_id = str(uuid.uuid4())
        resp = self._session.post(
            f"{BASE_URL}/userapigateway/trading/{account_id}/order",
            headers=self._headers(),
            json={
                "orderId": order_id,
                "instrument": {"symbol": symbol, "type": "EQUITY"},
                "orderSide": side,
                "orderType": "MARKET",
                "amount": str(amount_usd),
                "expiration": {"timeInForce": "DAY"},
            },
            timeout=self._timeout,
        )
        self._raise_for_status(resp, f"placing {side} order for {symbol}")
        return resp.json().get("orderId", order_id)

    def get_order(self, order_id: str) -> dict:
        account_id = self.resolve_account_id()
        resp = self._session.get(
            f"{BASE_URL}/userapigateway/trading/{account_id}/order/{order_id}",
            headers=self._headers(),
            timeout=self._timeout,
        )
        self._raise_for_status(resp, f"fetching order {order_id}")
        return resp.json()

    def wait_for_fill(self, order_id: str, timeout_s: int = 60, poll_s: int = 2) -> dict:
        """Polls an order until it reaches a terminal state or timeout_s
        elapses. Returns the final order payload; caller checks status."""
        terminal = {"FILLED", "CANCELLED", "REJECTED", "EXPIRED", "QUEUED_CANCELLED"}
        deadline = time.time() + timeout_s
        last = {}
        while time.time() < deadline:
            last = self.get_order(order_id)
            if last.get("status") in terminal:
                return last
            time.sleep(poll_s)
        return last

    def get_buying_power(self) -> float:
        """Cash-only buying power (not margin-inflated). Public.com grants
        this instantly the moment a SELL order fills — even before the
        cash formally settles T+1 — so callers use this as a single,
        immediate check right before firing BUY orders rather than
        polling/waiting for anything (see rebalance_core.execute_trades()'s
        Phase 2). A prior version of this client waited for buying power
        to "increase" from a pre-sell baseline, which was actively wrong
        on a margin account: buying power can already be sufficient
        *before* the SELL even executes, so it would never "increase" and
        the wait would spuriously time out."""
        raw = self.get_portfolio()
        buying_power = raw.get("buyingPower")
        if isinstance(buying_power, dict):
            for key in ("cashOnlyBuyingPower", "buyingPower"):
                val = buying_power.get(key)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        continue
        return parse_portfolio(raw)["cash"]
