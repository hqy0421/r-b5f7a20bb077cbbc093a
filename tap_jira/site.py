"""Stable site identity resolution for multi-instance deployments.

Two independent Jira Cloud instances may expose identical issue and project
IDs, so records from different sites cannot be distinguished by their native
keys alone. This module derives a *stable* site identifier that:

* is independent of the access alias used for the request (a site may be
  reached through several hostnames / base URLs that all return the same
  data), and
* is safe to use as a record field, a component of a composite primary key
  and as a state-partition key.

Resolution order (first stable value wins):

1. The configured ``cloud_id``. A Cloud ID is an Atlassian-issued tenant
   identifier and never changes when the site is accessed through an alias.
2. The canonical ``baseUrl`` reported by the site itself via the
   ``/serverInfo`` endpoint, reduced to its normalized hostname. The site's
   own answer is authoritative regardless of the hostname used to call it.
3. The normalized configured ``domain`` as a last resort, so identity stays
   deterministic even when the identifying endpoint cannot be reached.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import requests
import requests.auth

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

#: Name of the field carrying the stable site identifier on every record.
SITE_ID_KEY = "site_id"

_SERVER_INFO_PATH = "/rest/api/3/serverInfo"
_REQUEST_TIMEOUT = 30


def normalize_host(value: str) -> str:
    """Return a stable, filesystem- and key-safe representation of a host.

    Accepts either a bare hostname or a full URL. Scheme, path, query,
    username/password and an explicit port are stripped, the value is
    lower-cased and dots are replaced with underscores.

    >>> normalize_host("https://Example.Atlassian.net:443/jira")
    'example_atlassian_net'
    >>> normalize_host("example.atlassian.net")
    'example_atlassian_net'
    """
    candidate = value.strip()
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    parsed = urlparse(candidate)
    hostname = (parsed.hostname or candidate).lower()
    return hostname.replace(".", "_")


def _server_info_url(domain: str) -> str:
    base = domain.strip()
    if "://" not in base:
        base = f"https://{base}"
    return f"{base.rstrip('/')}{_SERVER_INFO_PATH}"


def fetch_canonical_base_url(
    config: Mapping[str, Any],
    *,
    session: requests.Session | None = None,
) -> str | None:
    """Ask the site for its canonical base URL via ``/serverInfo``.

    Returns ``None`` (and logs a warning) when the endpoint cannot be reached
    or the response does not contain a usable ``baseUrl``. Alias hostnames do
    not matter here: the value returned is the site's own configured base URL.
    """
    domain = config["domain"]
    http = session or requests.Session()
    try:
        response = http.get(
            _server_info_url(domain),
            auth=requests.auth.HTTPBasicAuth(
                username=config["email"],
                password=config["api_token"],
            ),
            headers={
                "Accept": "application/json",
                "Accept-Language": "en",
                "X-Force-Accept-Language": "true",
            },
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        base_url = response.json().get("baseUrl")
    except (requests.RequestException, ValueError):
        logger.warning(
            "Unable to resolve canonical base URL from /serverInfo for %s; "
            "falling back to configured domain for the site identity.",
            domain,
            exc_info=True,
        )
        return None

    if not isinstance(base_url, str) or not base_url.strip():
        logger.warning(
            "/serverInfo for %s did not return a usable baseUrl; falling back "
            "to configured domain for the site identity.",
            domain,
        )
        return None

    return base_url


@functools.cache
def _site_id_from_canonical_base_url(canonical_base_url: str) -> str:
    return normalize_host(canonical_base_url)


def resolve_site_id(
    *,
    cloud_id: str | None,
    domain: str,
    email: str,
    api_token: str,
) -> str:
    """Resolve the stable site identifier for a given tap configuration.

    The network lookup is cached by the immutable configuration tuple so that
    all streams of a single tap invocation share one identity and the
    ``/serverInfo`` endpoint is hit at most once. The cache is keyed on the
    full configuration (including domain), so two sites configured with
    different domains in the same process never share an identity.

    Args:
        cloud_id: Optional Atlassian Cloud ID from configuration.
        domain: Configured site domain.
        email: API user email (used for the identifying request).
        api_token: API token (used for the identifying request).

    Returns:
        A stable, key-safe site identifier.
    """
    if cloud_id:
        # The Cloud ID is a tenant identifier issued by Atlassian and is
        # unaffected by any alias used to reach the site.
        return normalize_host(cloud_id)

    canonical_base_url = fetch_canonical_base_url(
        {
            "domain": domain,
            "email": email,
            "api_token": api_token,
        },
    )
    if canonical_base_url:
        return _site_id_from_canonical_base_url(canonical_base_url)

    return normalize_host(domain)
