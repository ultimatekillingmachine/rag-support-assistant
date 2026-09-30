"""Access control: who is asking, and what may they see.

Threat model, stated plainly: the API is reachable by customers. Internal
articles (``audience: operator``) contain escalation rules and compensation
limits. If the *client* told us which articles to search, anyone could pass
``audience=operator`` and read them. So the role is derived on the server from
a bearer token, and the request body may only express intent (a question), not
authorisation.

Tokens are read from settings, which in turn come from the environment, so no
secret is committed to the repository. Comparison uses ``secrets.compare_digest``
to avoid leaking token contents through response timing.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from fastapi import Header, HTTPException

from app.config import Settings
from app.config import settings as default_settings

ROLE_CUSTOMER = "customer"
ROLE_OPERATOR = "operator"

# Which article audience each role is allowed to retrieve.
# ``None`` means "no filter" — the role sees public and internal articles.
ROLE_AUDIENCE_FILTER: dict[str, str | None] = {
    ROLE_CUSTOMER: "customer",
    ROLE_OPERATOR: None,
}


@dataclass(frozen=True)
class Principal:
    """An authenticated caller: a role plus the token it presented."""

    role: str
    token: str

    @property
    def audience_filter(self) -> str | None:
        """Audience value to pass into retrieval (``None`` = no restriction)."""
        return ROLE_AUDIENCE_FILTER[self.role]


def _iter_tokens(config: Settings):
    yield ROLE_CUSTOMER, config.api_token_customer
    yield ROLE_OPERATOR, config.api_token_operator


def authenticate(config: Settings | None = None, token: str | None = None) -> Principal | None:
    """Resolve a bearer token to a principal, or ``None`` when unauthorised.

    When ``auth_enabled`` is false the caller is treated as an operator-level
    principal without a token — convenient for local experiments, never for a
    deployment that holds internal content.
    """
    config = config or default_settings

    if not config.auth_enabled:
        return Principal(role=ROLE_OPERATOR, token="")

    if not token:
        return None
    for role, expected in _iter_tokens(config):
        if expected and secrets.compare_digest(token, expected):
            return Principal(role=role, token=token)
    return None


def require_principal(
    authorization: str | None = Header(default=None),
) -> Principal:
    """FastAPI dependency: turn the Authorization header into a principal.

    Declared as an explicit ``Header`` parameter (not ``Annotated``) because a
    dependency defined inside a closure is not recognised by FastAPI and would
    otherwise be treated as a query parameter.
    """
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    principal = authenticate(default_settings, token)
    if principal is None:
        raise HTTPException(
            status_code=401,
            detail="Provide a valid token: Authorization: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return principal
