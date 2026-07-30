"""Public output contract for split-produced records (D9/D14).

No internal split/reduction content (division, documentation_units, unit
anchors, coverage claims) is ever part of the published file-level shape —
only the ordinary documentation fields plus the private `_large_file_identity`
survive into JSON, Markdown, and their round trips.
"""

from __future__ import annotations

import json

from codedoc.core.document import read_codedoc_document
from codedoc.core.markdown_view import (
    json_from_markdown,
    markdown_from_view,
    markdown_to_view,
    read_embedded_view,
)
from codedoc.core.project_view import (
    build_project_view,
    clean_file_record,
    json_from_view,
)
from codedoc.core.resume import _public_record_to_doc
from codedoc.core.result_assembly import (
    MAX_PUBLIC_EXPORT_ITEM_CHARS,
    MAX_PUBLIC_EXPORT_ITEMS,
    MAX_PUBLIC_SYMBOL_ITEMS_PER_KIND,
    MAX_PUBLIC_SYMBOL_NAME_CHARS,
)
from tests.support.run_metadata_cases import _split_record, _split_stats


def _raw_record_with_stray_internal_keys() -> dict:
    """A raw pipeline record whose documentation dict carries stray
    division/documentation_units keys, as if predecessor-shaped data had
    slipped through. clean_file_record must discard them unconditionally —
    it never special-cases their presence, it simply never reads them."""
    return {
        "hash": "abc",
        "file_path": "src/large.py",
        "language": "python",
        "_large_file_identity": "large-file-v2:abc",
        "documentation": {
            "file_path": "src/large.py",
            "language": "python",
            "description": "Large file.",
            "functions": [{"name": "load", "description": "Loads values."}],
            "documentation_units": [{"unit_id": "unit_" + "a" * 64, "title": "x"}],
            "division": {"strategy": "split", "unit_count": 2, "chunk_count": 3},
            "state": "forged",
            "relationships": [{"from": "x", "to": "y"}],
            "planner": {"task": "forged"},
        },
    }


def test_clean_file_record_never_emits_division_or_documentation_units() -> None:
    record = clean_file_record(_raw_record_with_stray_internal_keys())

    assert "division" not in record
    assert "documentation_units" not in record
    assert record["description"] == "Large file."
    assert record["functions"] == [{"name": "load", "description": "Loads values."}]
    assert record["_large_file_identity"] == "large-file-v2:abc"
    assert not {"state", "relationships", "planner"} & set(record)


def test_split_produced_record_round_trips_and_is_a_defensive_copy() -> None:
    raw = _raw_record_with_stray_internal_keys()
    record = clean_file_record(raw)

    # Mutating the source after cleaning must not affect the cleaned record.
    raw["documentation"]["description"] = "mutated"
    assert record["description"] == "Large file."

    restored = _public_record_to_doc(record)
    record["description"] = "changed later"
    assert restored["description"] == "Large file."
    assert "division" not in restored
    assert "documentation_units" not in restored
    assert restored["_large_file_identity"] == "large-file-v2:abc"


def test_clean_file_record_projects_internal_facts_through_public_limits() -> None:
    raw = _raw_record_with_stray_internal_keys()
    raw["documentation"]["functions"] = [
        {
            "name": f"function_{index}",
            "description": f"Function {index}.",
            "signature": f"function_{index}()",
            "_provenance": [{"source_order": index}],
        }
        for index in range(MAX_PUBLIC_SYMBOL_ITEMS_PER_KIND + 4)
    ]
    raw["documentation"]["classes"] = [
        {
            "name": f"Class{index}",
            "signature": f"class Class{index}",
            "_provenance": [{"source_order": index}],
        }
        for index in range(MAX_PUBLIC_SYMBOL_ITEMS_PER_KIND + 2)
    ]
    long_export = "e" * 200
    raw["documentation"]["exports"] = [
        long_export,
        *(
            f"export_{index}"
            for index in range(MAX_PUBLIC_EXPORT_ITEMS + 4)
        ),
    ]

    record = clean_file_record(raw)

    assert len(record["functions"]) == MAX_PUBLIC_SYMBOL_ITEMS_PER_KIND
    assert len(record["classes"]) == MAX_PUBLIC_SYMBOL_ITEMS_PER_KIND
    assert len(record["exports"]) == MAX_PUBLIC_EXPORT_ITEMS
    assert all(
        set(item) <= {"name", "description"}
        for item in record["functions"] + record["classes"]
    )
    assert record["exports"][0] == long_export
    assert len(record["exports"][0]) > MAX_PUBLIC_SYMBOL_NAME_CHARS
    assert record["_large_file_identity"] == "large-file-v2:abc"


def test_split_last_run_stats_round_trip_as_valid_json(tmp_path) -> None:
    split_view = build_project_view([_split_record()], _split_stats())
    path = tmp_path / "codedoc.json"
    path.write_text(json_from_view(split_view), encoding="utf-8")

    document = read_codedoc_document(path)
    assert document.view["last_run"]["split_chunks"] == 2
    assert document.view["last_run"]["unit_documentation_calls_planned"] == 2
    assert document.view["last_run"]["split_final_synthesis_calls_planned"] == 1
    assert "division" not in document.files[0]
    assert "documentation_units" not in document.files[0]


def test_markdown_round_trips_split_stats_losslessly() -> None:
    view = build_project_view([_split_record()], _split_stats())
    markdown = markdown_from_view(view)
    assert markdown_to_view(markdown) == view


def test_prefixed_internal_facts_are_projected_on_every_public_conversion(
    tmp_path,
) -> None:
    view = build_project_view([_split_record()], _split_stats())
    file_record = view["files"][0]
    file_record["functions"] = [
        {
            "name": f"function_{index}_" + "n" * 160,
            "description": f"Function {index}.",
            "signature": f"function_{index}()",
            "_provenance": [{"chunk_id": "chunk_" + f"{index:064x}"}],
        }
        for index in range(MAX_PUBLIC_SYMBOL_ITEMS_PER_KIND + 5)
    ]
    file_record["classes"] = [
        {
            "name": f"Class{index}",
            "signature": f"class Class{index}",
            "_provenance": [{"source_order": index}],
        }
        for index in range(MAX_PUBLIC_SYMBOL_ITEMS_PER_KIND + 3)
    ]
    file_record["exports"] = [
        f"export_{index}_" + "e" * 160
        for index in range(MAX_PUBLIC_EXPORT_ITEMS + 7)
    ]

    markdown = markdown_from_view(view)
    markdown_path = tmp_path / "codedoc.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    payloads = (
        json.loads(json_from_view(view)),
        read_embedded_view(markdown),
        markdown_to_view(markdown),
        json.loads(json_from_markdown(markdown)),
        read_codedoc_document(markdown_path).view,
    )

    for payload in payloads:
        projected = payload["files"][0]
        assert len(projected["functions"]) == MAX_PUBLIC_SYMBOL_ITEMS_PER_KIND
        assert len(projected["classes"]) == MAX_PUBLIC_SYMBOL_ITEMS_PER_KIND
        assert len(projected["exports"]) == MAX_PUBLIC_EXPORT_ITEMS
        assert all(
            set(item) <= {"name", "description"}
            for item in projected["functions"] + projected["classes"]
        )
        assert all(
            len(item["name"]) <= MAX_PUBLIC_SYMBOL_NAME_CHARS
            for item in projected["functions"] + projected["classes"]
        )
        assert all(
            len(export) <= MAX_PUBLIC_EXPORT_ITEM_CHARS
            for export in projected["exports"]
        )
        assert len(projected["exports"][0]) > MAX_PUBLIC_SYMBOL_NAME_CHARS
        assert projected["_large_file_identity"] == "large-file-v2:test"


def test_ordinary_truncate_view_has_no_split_surface() -> None:
    ordinary_view = build_project_view(
        [
            {
                "hash": "small",
                "file_path": "small.py",
                "language": "python",
                "documentation": {"description": "Small file."},
            }
        ],
        {"checked": 1},
    )
    ordinary_json = json.loads(json_from_view(ordinary_view))
    ordinary_markdown = markdown_from_view(ordinary_view)
    assert "large_file_strategy" not in ordinary_json["last_run"]
    assert not any(key.startswith("split_") for key in ordinary_json["last_run"])
    assert "Large-file split" not in ordinary_markdown
    assert "Source coverage" not in ordinary_markdown
