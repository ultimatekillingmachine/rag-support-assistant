"""Tests for authentication and role-based article visibility."""

from __future__ import annotations

from app.auth import (
    ROLE_AUDIENCE_FILTER,
    ROLE_CUSTOMER,
    ROLE_OPERATOR,
    Principal,
    authenticate,
)
from app.config import Settings


def _config(**overrides) -> Settings:
    base = {
        "auth_enabled": True,
        "api_token_customer": "customer-token",
        "api_token_operator": "operator-token",
    }
    base.update(overrides)
    return Settings(**base)


def test_valid_customer_token_maps_to_customer_role() -> None:
    principal = authenticate(_config(), "customer-token")
    assert principal is not None
    assert principal.role == ROLE_CUSTOMER
    assert principal.audience_filter == "customer"


def test_valid_operator_token_maps_to_operator_role() -> None:
    principal = authenticate(_config(), "operator-token")
    assert principal is not None
    assert principal.role == ROLE_OPERATOR
    assert principal.audience_filter is None  # no restriction: seeded + internal


def test_unknown_or_missing_token_is_rejected() -> None:
    config = _config()
    assert authenticate(config, "super-secret") is None
    assert authenticate(config, None) is None
    assert authenticate(config, "") is None


def test_empty_configured_token_never_authenticates() -> None:
    # An empty token in configuration must not become a bypass.
    config = _config(api_token_customer="")
    assert authenticate(config, "") is None


def test_auth_disabled_returns_operator_principal() -> None:
    principal = authenticate(_config(auth_enabled=False), None)
    assert principal is not None
    assert principal.role == ROLE_OPERATOR


def test_role_audience_mapping_is_explicit() -> None:
    assert ROLE_AUDIENCE_FILTER == {"customer": "customer", "operator": None}
    assert Principal(role=ROLE_CUSTOMER, token="x").audience_filter == "customer"
    assert Principal(role=ROLE_OPERATOR, token="x").audience_filter is None
