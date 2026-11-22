"""Tests for stable site identity resolution."""

from __future__ import annotations

from http import HTTPStatus
from typing import Any
from unittest.mock import Mock, patch

import pytest
import requests

from tap_jira.site import (
    SITE_ID_KEY,
    fetch_canonical_base_url,
    normalize_host,
    resolve_site_id,
)

# Placeholder credential used only to exercise request construction.
API_TOKEN = "token"
TEST_EMAIL = "user@example.com"


def _mock_response(
    status_code: int = 200,
    payload: dict[str, Any] | None = None,
) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = status_code
    response.json.return_value = payload or {}
    if status_code >= HTTPStatus.BAD_REQUEST:
        response.raise_for_status.side_effect = requests.HTTPError(
            f"{status_code} error",
            response=response,
        )
    else:
        response.raise_for_status.return_value = None
    return response


class TestNormalizeHost:
    """Host normalization must be alias- and presentation-independent."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("example.atlassian.net", "example_atlassian_net"),
            ("https://example.atlassian.net", "example_atlassian_net"),
            ("https://Example.ATLASSIAN.net:443/jira", "example_atlassian_net"),
            ("http://alias.example.com/path?x=1", "alias_example_com"),
            ("https://user:pass@host.example.net/", "host_example_net"),
        ],
    )
    def test_normalize_host(self, value: str, expected: str) -> None:
        assert normalize_host(value) == expected

    def test_normalize_host_strips_whitespace(self) -> None:
        assert normalize_host("  example.atlassian.net  ") == "example_atlassian_net"


class TestFetchCanonicalBaseUrl:
    """The site's own baseUrl wins over the hostname used to call it."""

    def test_returns_canonical_base_url_from_server_info(self) -> None:
        response = _mock_response(
            payload={
                "baseUrl": "https://canonical.example.atlassian.net",
                "version": "1000.0.0",
            },
        )

        session = requests.Session()
        with patch.object(session, "get", return_value=response) as mocked_get:
            result = fetch_canonical_base_url(
                {
                    "domain": "alias.example.net",
                    "email": TEST_EMAIL,
                    "api_token": API_TOKEN,
                },
                session=session,
            )

        assert result == "https://canonical.example.atlassian.net"
        requested_url = mocked_get.call_args.args[0]
        # The request goes to the configured alias, but the returned identity
        # is derived from the site-reported canonical URL.
        assert requested_url == "https://alias.example.net/rest/api/3/serverInfo"

    def test_aliases_share_identity_when_site_reports_same_base_url(self) -> None:
        response = _mock_response(
            payload={"baseUrl": "https://real.atlassian.net"},
        )

        for alias in ("alias-one.example.com", "alias-two.example.org"):
            session = requests.Session()
            with patch.object(session, "get", return_value=response):
                base_url = fetch_canonical_base_url(
                    {
                        "domain": alias,
                        "email": TEST_EMAIL,
                        "api_token": API_TOKEN,
                    },
                    session=session,
                )
            assert normalize_host(base_url or "") == "real_atlassian_net"

    def test_returns_none_on_connection_error(self) -> None:
        session = requests.Session()
        with patch.object(
            session,
            "get",
            side_effect=requests.ConnectionError("unreachable"),
        ):
            result = fetch_canonical_base_url(
                {
                    "domain": "offline.atlassian.net",
                    "email": TEST_EMAIL,
                    "api_token": API_TOKEN,
                },
                session=session,
            )
        assert result is None

    def test_returns_none_on_http_error(self) -> None:
        response = _mock_response(status_code=401)
        session = requests.Session()
        with patch.object(session, "get", return_value=response):
            result = fetch_canonical_base_url(
                {
                    "domain": "denied.atlassian.net",
                    "email": TEST_EMAIL,
                    "api_token": API_TOKEN,
                },
                session=session,
            )
        assert result is None

    def test_returns_none_when_base_url_missing(self) -> None:
        response = _mock_response(payload={"version": "1000.0.0"})
        session = requests.Session()
        with patch.object(session, "get", return_value=response):
            result = fetch_canonical_base_url(
                {
                    "domain": "weird.atlassian.net",
                    "email": TEST_EMAIL,
                    "api_token": API_TOKEN,
                },
                session=session,
            )
        assert result is None


class TestResolveSiteId:
    """Resolution order and cross-alias stability of the site identifier."""

    def test_cloud_id_takes_precedence(self) -> None:
        with patch(
            "tap_jira.site.fetch_canonical_base_url",
            side_effect=AssertionError("serverInfo must not be called"),
        ):
            site_id = resolve_site_id(
                cloud_id="11223344-a1b2",
                domain="whatever.atlassian.net",
                email=TEST_EMAIL,
                api_token=API_TOKEN,
            )
        assert site_id == "11223344-a1b2".replace(".", "_")

    def test_canonical_base_url_used_without_cloud_id(self) -> None:
        with patch(
            "tap_jira.site.fetch_canonical_base_url",
            return_value="https://real.atlassian.net",
        ):
            site_id = resolve_site_id(
                cloud_id=None,
                domain="alias.example.com",
                email=TEST_EMAIL,
                api_token=API_TOKEN,
            )
        assert site_id == "real_atlassian_net"

    def test_alias_domains_resolve_to_same_site_id(self) -> None:
        with patch(
            "tap_jira.site.fetch_canonical_base_url",
            return_value="https://real.atlassian.net",
        ):
            first = resolve_site_id(
                cloud_id=None,
                domain="alias-one.example.com",
                email=TEST_EMAIL,
                api_token=API_TOKEN,
            )
            second = resolve_site_id(
                cloud_id=None,
                domain="alias-two.example.org",
                email=TEST_EMAIL,
                api_token=API_TOKEN,
            )
        assert first == second == "real_atlassian_net"

    def test_falls_back_to_domain_when_endpoint_unavailable(self) -> None:
        with patch(
            "tap_jira.site.fetch_canonical_base_url",
            return_value=None,
        ):
            site_id = resolve_site_id(
                cloud_id=None,
                domain="fallback.atlassian.net",
                email=TEST_EMAIL,
                api_token=API_TOKEN,
            )
        assert site_id == "fallback_atlassian_net"

    def test_two_distinct_sites_get_distinct_ids(self) -> None:
        with patch(
            "tap_jira.site.fetch_canonical_base_url",
            side_effect=lambda config: (
                "https://alpha.atlassian.net"
                if "alpha" in config["domain"]
                else "https://beta.atlassian.net"
            ),
        ):
            alpha = resolve_site_id(
                cloud_id=None,
                domain="alpha.atlassian.net",
                email=TEST_EMAIL,
                api_token=API_TOKEN,
            )
            beta = resolve_site_id(
                cloud_id=None,
                domain="beta.atlassian.net",
                email=TEST_EMAIL,
                api_token=API_TOKEN,
            )
        assert alpha != beta

    def test_site_id_key_constant(self) -> None:
        assert SITE_ID_KEY == "site_id"
