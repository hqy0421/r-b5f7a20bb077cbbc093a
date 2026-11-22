"""REST client handling, including tap-jiraStream base class."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, TypeVar, cast

import requests.auth
from singer_sdk import typing as th
from singer_sdk.streams import RESTStream

from tap_jira.site import SITE_ID_KEY

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from requests import Response
    from singer_sdk.helpers.types import Context

    from tap_jira.tap import TapJira

    _Auth = Callable[[requests.PreparedRequest], requests.PreparedRequest]


_TNextPageToken = TypeVar("_TNextPageToken")


class ResumableAPIError(Exception):
    """Exception raised for resumable API errors."""

    @override
    def __init__(self, message: str, response: requests.Response) -> None:
        super().__init__(message)
        self.response = response


class JiraStream(RESTStream[_TNextPageToken]):
    """tap-jira stream class."""

    next_page_token_jsonpath = "$.paging.start"  # noqa: S105
    records_jsonpath = "$[*]"  # Or override `parse_response`.
    instance_name: str

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Rewrite native key declarations into site-aware class attributes.

        Subclasses declare ``primary_keys`` / ``state_partitioning_keys`` as
        plain class attributes. A data descriptor on a parent class (such as
        a property) cannot win attribute lookup against a plain attribute on
        the subclass itself, so overriding these as properties would be
        silently shadowed. Instead each subclass declaration is rewritten at
        class-creation time to prepend :data:`~tap_jira.site.SITE_ID_KEY`,
        which keeps both the SDK constructor and catalog-driven
        :meth:`~singer_sdk.Stream.apply_catalog` assignments site-aware.
        """
        super().__init_subclass__(**kwargs)

        native_primary_keys = cls.__dict__.get("primary_keys")
        if (
            isinstance(native_primary_keys, tuple)
            and SITE_ID_KEY not in native_primary_keys
        ):
            cls.primary_keys = (  # type: ignore[method-assign,assignment]
                SITE_ID_KEY,
                *native_primary_keys,
            )

        native_partition_keys = cls.__dict__.get("state_partitioning_keys")
        if native_partition_keys is None:
            cls.state_partitioning_keys = (  # type: ignore[method-assign,assignment]
                SITE_ID_KEY,
            )
        elif SITE_ID_KEY not in native_partition_keys:
            cls.state_partitioning_keys = (  # type: ignore[method-assign,assignment]
                SITE_ID_KEY,
                *native_partition_keys,
            )

        native_schema = cls.__dict__.get("schema")
        if isinstance(native_schema, dict):
            properties = native_schema.setdefault("properties", {})
            if SITE_ID_KEY not in properties:
                native_schema["properties"] = {
                    SITE_ID_KEY: th.StringType().to_dict(),
                    **properties,
                }

    @property
    def site_id(self) -> str:
        """Stable identity of the Jira site this stream reads from."""
        return cast("TapJira", self._tap).site_id

    @override
    @property
    def schema(self) -> dict[str, Any]:
        """Stream schema with the stable site identity field added.

        The field is injected centrally so every master and child record can
        be attributed to its source site, including streams whose schemas are
        built dynamically.
        """
        stream_schema = cast("dict[str, Any]", super().schema)
        properties = stream_schema.setdefault("properties", {})
        if SITE_ID_KEY not in properties:
            stream_schema["properties"] = {
                SITE_ID_KEY: th.StringType().to_dict(),
                **properties,
            }
        return stream_schema

    @override
    @property
    def partitions(self) -> list[dict[str, Any]] | None:
        """Drive top-level syncs through the site-scoped state partition.

        Parent streams are invoked without a context; returning the site
        partition here makes bookmark reads and writes land in the same
        partition as child streams, so two sites sharing one state file keep
        fully isolated incremental progress. Partitions already recorded in
        state are preserved for bookmark lookups.
        """
        existing = super().partitions or []
        site_partition = {SITE_ID_KEY: self.site_id}
        if site_partition not in existing:
            return [site_partition, *existing]
        return existing

    @override
    def post_process(
        self,
        row: dict[str, Any],
        context: Context | None = None,
    ) -> dict[str, Any] | None:
        """Stamp every record with the stable site identifier."""
        row[SITE_ID_KEY] = self.site_id
        return row

    @override
    def get_child_context(
        self,
        record: dict[str, Any],
        context: Context | None,
    ) -> dict[str, Any] | None:
        """Propagate the site identity into child-stream contexts."""
        child_context = super().get_child_context(record, context)
        if child_context is not None:
            context_dict = cast("dict[str, Any]", child_context)
            context_dict[SITE_ID_KEY] = self.site_id
            return context_dict
        return None

    @override
    @property
    def url_base(self) -> str:
        """Returns base url."""
        cloud_id = self.config.get("cloud_id")
        if cloud_id:
            return f"https://api.atlassian.com/ex/jira/{cloud_id}/rest/api/3"
        domain = self.config["domain"]
        return f"https://{domain}/rest/api/3"

    @override
    @property
    def authenticator(self) -> _Auth:
        """Stream authenticator."""
        return requests.auth.HTTPBasicAuth(
            password=self.config["api_token"],
            username=self.config["email"],
        )

    @override
    @property
    def http_headers(self) -> dict[str, str]:
        """Force English-language responses from the Jira Cloud API.

        Without this, error messages (e.g. "The board does not support
        sprints") are localized to the API user's profile language, which
        breaks string-matching in ``validate_response`` overrides.
        See https://developer.atlassian.com/cloud/jira/platform/rest/v3/intro/#special-request-headers
        """
        headers: dict[str, str] = super().http_headers
        headers["Accept-Language"] = "en"
        headers["X-Force-Accept-Language"] = "true"
        return headers

    @override
    def get_records(self, context: Context | None) -> Iterable[dict[str, Any]]:
        try:
            yield from super().get_records(context)
        except ResumableAPIError as e:
            self.logger.warning(e)


class JiraStartAtPaginatedStream(JiraStream[int]):
    """Jira stream that uses the startAt pagination parameter."""

    @override
    def get_url_params(
        self,
        context: Context | None,
        next_page_token: int | None,
    ) -> dict[str, Any]:
        """Return a dictionary of values to be used in URL parameterization."""
        params: dict[str, Any] = {}
        if next_page_token:
            params["startAt"] = next_page_token
        if self.replication_key:
            params["sort"] = "asc"
            params["order_by"] = self.replication_key

        return params

    def get_next_page_token(
        self,
        response: Response,
        previous_token: int | None,
    ) -> int | None:
        """Return a token for identifying next page or None if no more pages."""
        # If pagination is required, return a token which can be used to get the
        #       next page. If this is the final page, return "None" to end the
        #       pagination loop.
        resp_json = response.json()

        if previous_token is None:
            previous_token = 0

        total = -1
        results = 0
        _value = None
        is_last = None

        if (
            isinstance(resp_json, dict)
            and resp_json.get(self.instance_name) is not None
        ):
            _value = resp_json.get(self.instance_name)
            total = resp_json.get("total", -1)
            is_last = resp_json.get("isLast")
            results = len(_value)  # type: ignore[arg-type]

        if isinstance(is_last, bool) and total == -1 and not is_last:
            return previous_token + results

        if _value is None:
            page = resp_json
            if len(page) == 0 or total <= previous_token + results:
                return None
        elif len(_value) == 0 or total <= previous_token + results:
            return None

        return previous_token + results
