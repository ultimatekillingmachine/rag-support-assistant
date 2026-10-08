"""CLI: ``python -m scripts.smoke_api [base_url]`` — end-to-end check against a running server.

Verifies the public contract with real HTTP calls:

* no token -> 401
* customer token -> answer with public sources only
* operator token -> the same question reaches internal articles
* ``/ask/stream`` -> server-sent events (sources, several tokens, done)
* chat page and its script are served

Kept as a script so it can be run by hand after a deployment (and used as the
deployment checklist in the README).
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

QUESTION = "регламент эскалации обращений операторов"


def call(base_url: str, token: str | None, path: str = "/ask") -> tuple[int, dict]:
    payload = json.dumps({"question": QUESTION}).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310 - fixed local URL
        f"{base_url}{path}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, {}


def stream_events(base_url: str, token: str, question: str) -> list[dict]:
    """Read the whole SSE response and parse it into events."""
    payload = json.dumps({"question": question}).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310 - fixed local URL
        f"{base_url}/ask/stream",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    events: list[dict] = []
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        assert response.headers["content-type"].startswith("text/event-stream")
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:") :].strip()))
    return events


def fetch(base_url: str, path: str = "/") -> str:
    with urllib.request.urlopen(f"{base_url}{path}", timeout=30) as response:  # noqa: S310
        return response.read().decode("utf-8")


def main(base_url: str) -> None:
    status, _ = call(base_url, None)
    print(f"no token       -> HTTP {status}")
    assert status == 401, "anonymous request must be rejected"

    status, body = call(base_url, "customer-token")
    assert status == 200, f"customer request failed: {status}"
    customer_slugs = [item["slug"] for item in body["sources"]]
    print(f"customer token -> HTTP {status} role={body['role']} sources={customer_slugs}")
    assert "internal-escalation" not in customer_slugs, "internal article leaked to customer"

    status, body = call(base_url, "operator-token")
    assert status == 200, f"operator request failed: {status}"
    operator_slugs = [item["slug"] for item in body["sources"]]
    print(f"operator token -> HTTP {status} role={body['role']} sources={operator_slugs}")
    assert "internal-escalation" in operator_slugs, "operator cannot reach internal article"

    events = stream_events(base_url, "customer-token", "Сколько идёт доставка в Москву?")
    kinds = [event["type"] for event in events]
    tokens = [event for event in events if event["type"] == "token"]
    print(f"stream         -> {len(events)} events, kinds={kinds[:4]}... tokens={len(tokens)}")
    assert kinds[0] == "sources", "sources must be sent first"
    assert kinds[-1] == "done", "stream must end with done"
    assert len(tokens) > 1, "the answer must arrive in several pieces"

    page = fetch(base_url, "/")
    script = fetch(base_url, "/static/app.js")
    print(f"chat page      -> {len(page)} bytes, script {len(script)} bytes")
    assert "Ассистент поддержки" in page
    assert "/ask/stream" in script

    print("smoke test: OK")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000")

