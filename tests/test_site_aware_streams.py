"""Tests for site-aware records, composite keys and site-partitioned state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from tap_jira.site import SITE_ID_KEY
from tap_jira.streams import (
    CustomFieldContextStream,
    FieldStream,
    IssueChangeLogStream,
    IssueComments,
    IssueStream,
    IssueWatchersStream,
    IssueWorklogs,
    ProjectStream,
    ProjectVersionStream,
    UsersStream,
)
from tap_jira.tap import TapJira

if TYPE_CHECKING:
    from singer_sdk.helpers.types import Context, Record


def _as_tuple(value: object) -> tuple[str, ...]:
    """Normalize an SDK key sequence for statically-checked assertions.

    The tap rewrites ``primary_keys`` / ``state_partitioning_keys`` at class
    creation time (see ``JiraStream.__init_subclass__``), which static type
    checkers cannot follow; accept the runtime value as ``object``.
    """
    if not value:
        return ()
    return tuple(value)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


ALPHA = "alpha_atlassian_net"
BETA = "beta_atlassian_net"
BASE_CONFIG = {
    "email": "user@example.com",
    "api_token": "token",
}


def _make_tap(domain: str, *, state: dict[str, Any] | None = None) -> TapJira:
    return TapJira(config={**BASE_CONFIG, "domain": domain}, state=state)


@pytest.fixture(autouse=True)
def _stub_site_resolution() -> Any:  # noqa: ANN401
    """Avoid real /serverInfo calls; identity derives from the test domain."""
    with patch(
        "tap_jira.tap.resolve_site_id",
        side_effect=lambda cloud_id, domain, email, api_token: domain.replace(  # noqa: ARG005
            ".",
            "_",
        ),
    ) as mocked:
        yield mocked


class TestSiteAwareSchemaAndKeys:
    """Every stream exposes site_id in schema, keys and state partitioning."""

    def test_schema_contains_site_id(self) -> None:
        tap = _make_tap("alpha.atlassian.net")
        for stream_cls in (
            UsersStream,
            FieldStream,
            ProjectStream,
            IssueStream,
            IssueComments,
        ):
            stream = stream_cls(tap)
            site_property = stream.schema["properties"][SITE_ID_KEY]
            # SDK typing may serialize as {"type": "string"} or
            # {"type": ["string"]}; accept both forms.
            assert "string" in (
                site_property["type"]
                if isinstance(site_property["type"], list)
                else [site_property["type"]]
            )

    def test_primary_keys_are_site_scoped(self) -> None:
        tap = _make_tap("alpha.atlassian.net")
        assert _as_tuple(UsersStream(tap).primary_keys) == (
            SITE_ID_KEY,
            "accountId",
        )
        assert _as_tuple(ProjectStream(tap).primary_keys) == (SITE_ID_KEY, "id")
        assert _as_tuple(IssueStream(tap).primary_keys) == (SITE_ID_KEY, "id")
        assert _as_tuple(IssueComments(tap).primary_keys) == (SITE_ID_KEY, "id")

    def test_state_partitioning_uses_site_id(self) -> None:
        tap = _make_tap("alpha.atlassian.net")
        # Top-level stream gets a default site partition.
        assert _as_tuple(ProjectStream(tap).state_partitioning_keys) == (SITE_ID_KEY,)
        # Child streams previously pinned to () now keep site partitioning.
        assert _as_tuple(IssueComments(tap).state_partitioning_keys) == (SITE_ID_KEY,)
        # Existing native partition keys are preserved alongside site_id.
        assert _as_tuple(CustomFieldContextStream(tap).state_partitioning_keys) == (
            SITE_ID_KEY,
            "fieldId",
        )

    def test_top_level_partition_is_the_site(self) -> None:
        tap = _make_tap("alpha.atlassian.net")
        assert ProjectStream(tap).partitions == [{SITE_ID_KEY: ALPHA}]


class TestRecordStamping:
    """post_process attributes every master and child record to its site."""

    @staticmethod
    def _issue_row(issue_id: str) -> Record:
        return {
            "id": issue_id,
            "key": "SHARED-1",
            "fields": {
                "created": "2026-01-01T00:00:00Z",
                "updated": "2026-01-02T00:00:00Z",
            },
        }

    def test_master_record_stamped(self) -> None:
        tap = _make_tap("alpha.atlassian.net")
        processed = IssueStream(tap).post_process(self._issue_row("10001"), None)
        assert processed is not None
        assert processed[SITE_ID_KEY] == ALPHA

    def test_child_record_stamped_with_parent_site(self) -> None:
        tap = _make_tap("alpha.atlassian.net")
        context: Context = {"issue_id": "10001", SITE_ID_KEY: ALPHA}
        row: Record = {"id": "c1", "body": {}}
        processed = IssueComments(tap).post_process(row, context)
        assert processed is not None
        assert processed[SITE_ID_KEY] == ALPHA
        assert processed["issueId"] == "10001"

    def test_same_native_id_distinct_sites_remain_distinguishable(self) -> None:
        alpha_tap = _make_tap("alpha.atlassian.net")
        beta_tap = _make_tap("beta.atlassian.net")

        alpha_row = IssueStream(alpha_tap).post_process(
            self._issue_row("10001"),
            None,
        )
        beta_row = IssueStream(beta_tap).post_process(
            self._issue_row("10001"),
            None,
        )

        assert alpha_row is not None
        assert beta_row is not None
        assert alpha_row["id"] == "10001"
        assert beta_row["id"] == "10001"
        assert alpha_row[SITE_ID_KEY] == ALPHA
        assert beta_row[SITE_ID_KEY] == BETA
        # Composite keys would not collide in the warehouse.
        alpha_composite = (alpha_row[SITE_ID_KEY], alpha_row["id"])
        beta_composite = (beta_row[SITE_ID_KEY], beta_row["id"])
        assert alpha_composite != beta_composite


class TestChildContextPropagation:
    """Site identity flows from parent records into child-stream contexts."""

    def test_issue_context_carries_site_id(self) -> None:
        tap = _make_tap("alpha.atlassian.net")
        context = IssueStream(tap).get_child_context(
            {"id": "10001"},
            {SITE_ID_KEY: ALPHA},
        )
        assert context == {"issue_id": "10001", SITE_ID_KEY: ALPHA}

    def test_project_context_carries_site_id(self) -> None:
        tap = _make_tap("alpha.atlassian.net")
        context = ProjectStream(tap).get_child_context(
            {"id": "77"},
            {SITE_ID_KEY: ALPHA},
        )
        assert context == {"projectId": "77", SITE_ID_KEY: ALPHA}

    def test_field_context_preserves_native_keys_and_site_id(self) -> None:
        tap = _make_tap("alpha.atlassian.net")
        record = {
            "id": "customfield_1",
            "custom": True,
            "schema": {"type": "option"},
        }
        context = FieldStream(tap).get_child_context(record, {SITE_ID_KEY: ALPHA})
        assert context == {
            "fieldId": "customfield_1",
            "supports_options": True,
            SITE_ID_KEY: ALPHA,
        }

    def test_all_issue_child_streams_partition_by_site(self) -> None:
        tap = _make_tap("alpha.atlassian.net")
        for child_cls in (
            IssueWatchersStream,
            IssueChangeLogStream,
            IssueComments,
            IssueWorklogs,
        ):
            stream = child_cls(tap)
            assert stream.parent_stream_type is IssueStream
            assert SITE_ID_KEY in _as_tuple(stream.state_partitioning_keys)


class TestStateIsolation:
    """Bookmarks from two sites sharing one state file never overwrite each other."""

    def test_bookmarks_partitioned_per_site(self) -> None:
        alpha_tap = _make_tap("alpha.atlassian.net")
        beta_tap = _make_tap("beta.atlassian.net")

        alpha_stream = IssueStream(alpha_tap)
        beta_stream = IssueStream(beta_tap)

        alpha_state = alpha_stream.get_context_state({SITE_ID_KEY: ALPHA})
        beta_state = beta_stream.get_context_state({SITE_ID_KEY: BETA})

        alpha_state["replication_key"] = "updated"
        alpha_state["replication_key_value"] = "2026-01-01T00:00:00Z"
        beta_state["replication_key"] = "updated"
        beta_state["replication_key_value"] = "2026-03-01T00:00:00Z"

        # Simulate two taps sharing one state artifact, as in a warehouse load.
        shared_state: dict[str, Any] = {
            "bookmarks": {
                "issues": {
                    "partitions": [
                        {
                            "context": {SITE_ID_KEY: ALPHA},
                            "replication_key": "updated",
                            "replication_key_value": "2026-01-01T00:00:00Z",
                        },
                        {
                            "context": {SITE_ID_KEY: BETA},
                            "replication_key": "updated",
                            "replication_key_value": "2026-03-01T00:00:00Z",
                        },
                    ],
                },
            },
        }
        merged_tap = _make_tap("alpha.atlassian.net", state=shared_state)
        merged_stream = IssueStream(merged_tap)

        partitions = merged_tap.state["bookmarks"]["issues"]["partitions"]
        partition_contexts = [p["context"] for p in partitions]
        assert {SITE_ID_KEY: ALPHA} in partition_contexts
        assert {SITE_ID_KEY: BETA} in partition_contexts

        # Re-reading each site returns its own bookmark, not the other's.
        assert (
            merged_stream.get_context_state({SITE_ID_KEY: ALPHA})[
                "replication_key_value"
            ]
            == "2026-01-01T00:00:00Z"
        )
        assert (
            merged_stream.get_context_state({SITE_ID_KEY: BETA})[
                "replication_key_value"
            ]
            == "2026-03-01T00:00:00Z"
        )

    def test_site_partition_preserved_when_state_already_has_other_site(
        self,
    ) -> None:
        state = {
            "bookmarks": {
                "projects": {
                    "partitions": [
                        {
                            "context": {SITE_ID_KEY: BETA},
                            "replication_key_value": "99",
                        },
                    ],
                },
            },
        }
        tap = _make_tap("alpha.atlassian.net", state=state)
        stream = ProjectStream(tap)
        partitions = stream.partitions or []
        assert {SITE_ID_KEY: ALPHA} in partitions
        assert {SITE_ID_KEY: BETA} in partitions


class TestEndToEndSync:
    """Record post-processing and site-partitioned state writes."""

    def test_child_record_gets_site_id_and_native_context(self) -> None:
        tap = _make_tap("alpha.atlassian.net")
        stream = ProjectVersionStream(tap)

        raw_record = {
            "id": "v1",
            "name": "1.0",
            "projectId": 77,
        }
        context = {SITE_ID_KEY: ALPHA, "projectId": "77"}

        processed = stream.post_process(dict(raw_record), context)
        assert processed is not None
        assert processed[SITE_ID_KEY] == ALPHA
        assert processed["id"] == "v1"

    def test_incremental_bookmark_lands_in_site_partition(self) -> None:
        tap = _make_tap("alpha.atlassian.net")
        stream = ProjectStream(tap)
        context = {SITE_ID_KEY: ALPHA}

        stream._increment_stream_state(  # noqa: SLF001
            {"id": "5", SITE_ID_KEY: ALPHA},
            context=context,
        )

        partition = stream.get_context_state(context)
        # Non-sorted streams keep advancing bookmarks under progress markers
        # until the partition is finalized.
        bookmark_location = partition.get("progress_markers", partition)
        assert bookmark_location["replication_key"] == "id"
        assert bookmark_location["replication_key_value"] == "5"
        assert tap.state["bookmarks"]["projects"]["partitions"][0]["context"] == context

    def test_full_sync_emits_stamped_records_with_composite_key(self) -> None:
        """Drive the real SDK sync loop and inspect emitted Singer messages."""
        tap = _make_tap("alpha.atlassian.net")
        stream = ProjectStream(tap)

        captured: list[dict[str, Any]] = []

        def _capture(message: Any) -> None:  # noqa: ANN401
            captured.append(message.to_dict())

        raw_records = [
            {"id": "77", "key": "ALPHA", "name": "Alpha project"},
            {"id": "88", "key": "BETA", "name": "Beta project"},
        ]

        with (
            patch.object(tap, "write_message", side_effect=_capture),
            patch.object(
                stream,
                "request_records",
                side_effect=lambda context: iter(raw_records),  # noqa: ARG005
            ),
        ):
            stream.sync(context={SITE_ID_KEY: ALPHA})

        record_messages = [m for m in captured if m["type"] == "RECORD"]
        schema_messages = [m for m in captured if m["type"] == "SCHEMA"]

        expected_record_count = 2
        assert len(record_messages) == expected_record_count
        assert all(m["record"][SITE_ID_KEY] == ALPHA for m in record_messages)
        assert {m["record"]["id"] for m in record_messages} == {"77", "88"}

        # The catalog metadata announces the composite primary key and the
        # injected site identity property.
        assert tuple(schema_messages[0]["key_properties"]) == (SITE_ID_KEY, "id")
        assert SITE_ID_KEY in schema_messages[0]["schema"]["properties"]

        # State landed in the site partition rather than a shared bookmark.
        project_state = tap.state["bookmarks"]["projects"]
        contexts = [p["context"] for p in project_state["partitions"]]
        assert {SITE_ID_KEY: ALPHA} in contexts
