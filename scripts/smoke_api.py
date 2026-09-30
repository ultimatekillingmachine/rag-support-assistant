"""CLI: ``python -m scripts.smoke_api [base_url]`` — end-to-end check against a running server.

Verifies the access-control contract with real HTTP calls:
* no token -> 401
* customer token -> answer with public sources only
* operator token -> the same question reaches internal articles

Kept as a script so it can be run by hand after a deployment.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

QUESTION = "регламент эскалации обращений операторов"


def call(base_url: str, token: str | None) -> tuple[int, dict]:
    payload = json.dumps({"question": QUESTION}).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310 - fixed local URL
        f"{base_url}/ask",
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


def main(base_url: str) -> None:
    status, _ = call(base_url, None)
    print(f"no token      -> HTTP {status}")
    assert status == 401, "anonymous request must be rejected"

    status, body = call(base_url, "customer-token")
    assert status == 200, f"customer request failed: {status}"
    customer_slugs = [item["slug"] for item in body["sources"]]
    print(f"customer token-> HTTP {status} role={body['role']} sources={customer_slugs}")
    assert "internal-escalation" not in customer_slugs, "internal article leaked to customer"

    status, body = call(base_url, "operator-token")
    assert status == 200, f"operator request failed: {status}"
    operator_slugs = [item["slug"] for item in body["sources"]]
    print(f"operator token-> HTTP {status} role={body['role']} sources={operator_slugs}")
    assert "internal-escalation" in operator_slugs, "operator cannot reach internal article"

    print("access control: OK")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000")
