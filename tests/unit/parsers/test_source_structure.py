from __future__ import annotations

import http.client
import socket
import sys
import urllib.request
from pathlib import Path

import pytest

import codedoc.parser.tree_sitter_structure as tree_sitter_structure
from codedoc.parser.language_specs import LANGUAGE_SPECS
from codedoc.parser.source_structure import (
    Atom,
    SourceRange,
    StructureResult,
    atom_id_for,
)
from codedoc.parser.tree_sitter_structure import (
    extract_structure,
    lexical_atoms,
    source_range_for_slice,
    validate_structure,
)
from tests.support.structure_extra import requires_structure_pack

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "large_file_division"

_DECLARATION_CASES = {
    "python": ("def alpha():\n    return 1\n", "alpha"),
    "javascript": ("function alpha() { return 1; }\n", "alpha"),
    "jsx": ("function Alpha() { return <div />; }\n", "Alpha"),
    "typescript": ("function alpha(): number { return 1; }\n", "alpha"),
    "tsx": ("function Alpha() { return <div />; }\n", "Alpha"),
    "dart": ("int alpha() { return 1; }\n", "alpha"),
    "java": ("class Alpha { int value() { return 1; } }\n", "Alpha"),
    "csharp": ("class Alpha { int Value() { return 1; } }\n", "Alpha"),
    "html": (
        "<html><body><script>function alpha() { return 1; }</script></body></html>\n",
        "alpha",
    ),
    "kotlin": ("fun alpha(): Int = 1\n", "alpha"),
    "swift": ("func alpha() -> Int { return 1 }\n", "alpha"),
    "go": ("package sample\nfunc alpha() int { return 1 }\n", "alpha"),
    "ruby": ("def alpha\n  1\nend\n", "alpha"),
    "rust": ("fn alpha() -> i32 { 1 }\n", "alpha"),
    "c": ("int alpha(void) { return 1; }\n", "alpha"),
    "cpp": ("int alpha() { return 1; }\n", "alpha"),
}


def _assert_complete(source: str, result: StructureResult) -> None:
    validate_structure(source, result)
    assert b"".join(
        atom.source.encode("utf-8") for atom in result.atoms
    ) == source.encode("utf-8")
    assert [atom.range.start_byte for atom in result.atoms] == [
        0,
        *[atom.range.end_byte for atom in result.atoms[:-1]],
    ]


def test_every_configured_language_has_a_real_declaration_fixture() -> None:
    assert set(_DECLARATION_CASES) == set(LANGUAGE_SPECS)


@requires_structure_pack
@pytest.mark.parametrize("language", sorted(_DECLARATION_CASES))
def test_configured_grammars_extract_symbols_and_complete_source(
    language: str,
) -> None:
    source, expected_name = _DECLARATION_CASES[language]

    result = extract_structure(f"src/example.{language}", language, source)

    assert result.structural_mode == "syntax", result.diagnostics
    _assert_complete(source, result)
    assert expected_name in {symbol.qualified_name for symbol in result.symbols}
    assert all(symbol.atom_id in {atom.atom_id for atom in result.atoms} for symbol in result.symbols)


@requires_structure_pack
@pytest.mark.parametrize(
    ("source", "expected_name", "expected_kind", "expected_signature"),
    [
        (
            "@trace\ndef sync_value():\n    return 1\n",
            "sync_value",
            "function_definition",
            "def sync_value():",
        ),
        (
            "@trace\r\n@tag('\u96ea')\r\n"
            "async def caf\u00e9():\r\n"
            "    return '\u96ea'\r\n",
            "caf\u00e9",
            "function_definition",
            "async def caf\u00e9():",
        ),
        (
            "@register('service')\nclass Service:\n    pass\n",
            "Service",
            "class_definition",
            "class Service:",
        ),
    ],
)
def test_python_decorated_definition_owns_every_decorator(
    source: str,
    expected_name: str,
    expected_kind: str,
    expected_signature: str,
) -> None:
    result = extract_structure("src/decorated.py", "python", source)

    assert result.structural_mode == "syntax", result.diagnostics
    _assert_complete(source, result)
    assert len(result.atoms) == 1
    symbol = next(
        symbol for symbol in result.symbols if symbol.qualified_name == expected_name
    )
    assert symbol.kind == expected_kind
    assert symbol.signature == expected_signature
    assert symbol.range.start_byte == 0
    assert result.atoms[0].source == source
    assert result.atoms[0].kind == expected_kind
    assert result.atoms[0].symbol_ids == (symbol.symbol_id,)


@requires_structure_pack
def test_html_fixture_keeps_host_bytes_and_remaps_inline_javascript_symbol() -> None:
    source = (FIXTURES / "inline_script.html").read_text(encoding="utf-8")

    result = extract_structure("web/index.html", "html", source)

    assert result.structural_mode == "syntax"
    _assert_complete(source, result)
    alpha = next(symbol for symbol in result.symbols if symbol.qualified_name == "alpha")
    encoded = source.encode("utf-8")
    assert encoded[alpha.range.start_byte : alpha.range.end_byte].decode("utf-8").startswith(
        "function alpha"
    )
    assert alpha.range.start_byte == encoded.index(b"function alpha")
    body_start = encoded.index(b"<script>") + len(b"<script>")
    body_end = encoded.index(b"</script>")
    assert all(
        atom.range.end_byte <= body_start or atom.range.start_byte >= body_start
        for atom in result.atoms
    )
    assert all(
        atom.range.end_byte <= body_end or atom.range.start_byte >= body_end
        for atom in result.atoms
    )
    embedded_atoms = [
        atom
        for atom in result.atoms
        if body_start <= atom.range.start_byte and atom.range.end_byte <= body_end
    ]
    assert "".join(atom.source for atom in embedded_atoms) == encoded[
        body_start:body_end
    ].decode("utf-8")
    assert next(
        atom for atom in embedded_atoms if "const before" in atom.source
    ).kind == "embedded_javascript_gap"
    assert next(
        atom for atom in embedded_atoms if "alpha();" in atom.source
    ).kind == "embedded_javascript_gap"


@requires_structure_pack
def test_html_inline_javascript_without_declarations_still_owns_every_body_byte() -> None:
    source = '<main>before</main><script>\nconst café = "雪";\ncafé++;\n</script><p>after</p>\n'
    encoded = source.encode("utf-8")

    result = extract_structure("web/unicode.html", "html", source)

    assert result.structural_mode == "syntax", result.diagnostics
    _assert_complete(source, result)
    body_start = encoded.index(b"<script>") + len(b"<script>")
    body_end = encoded.index(b"</script>")
    embedded_atoms = [
        atom
        for atom in result.atoms
        if body_start <= atom.range.start_byte and atom.range.end_byte <= body_end
    ]
    assert embedded_atoms
    assert all(atom.kind == "embedded_javascript_gap" for atom in embedded_atoms)
    assert b"".join(atom.source.encode("utf-8") for atom in embedded_atoms) == encoded[
        body_start:body_end
    ]


def test_missing_optional_package_falls_back_without_network_or_writes(
    tmp_path, monkeypatch
) -> None:
    fake_home = tmp_path / "home"
    fake_cache = tmp_path / "cache"
    fake_local = tmp_path / "local"
    fake_temp = tmp_path / "temp"
    for directory in (fake_home, fake_cache, fake_local, fake_temp):
        directory.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(fake_cache))
    monkeypatch.setenv("LOCALAPPDATA", str(fake_local))
    monkeypatch.setenv("TMP", str(fake_temp))
    monkeypatch.setenv("TEMP", str(fake_temp))

    def fail_network(*_args, **_kwargs):
        pytest.fail("structure fallback attempted network")

    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", None)
    monkeypatch.setattr(socket, "create_connection", fail_network)
    monkeypatch.setattr(socket.socket, "connect", fail_network)
    monkeypatch.setattr(urllib.request, "urlopen", fail_network)
    monkeypatch.setattr(http.client.HTTPConnection, "connect", fail_network)
    monkeypatch.setattr(http.client.HTTPSConnection, "connect", fail_network)

    def snapshot(directory: Path):
        return tuple(
            (
                path.relative_to(directory).as_posix(),
                path.is_dir(),
                0 if path.is_dir() else path.stat().st_size,
                path.stat().st_mtime_ns,
            )
            for path in sorted(directory.rglob("*"))
        )

    watched = (fake_home, fake_cache, fake_local, fake_temp)
    before = {directory: snapshot(directory) for directory in watched}

    result = extract_structure(
        "src/example.py",
        "python",
        "def alpha():\n    return 1\n",
    )

    assert result.structural_mode == "lexical"
    assert result.symbols == ()
    assert "optional bundled grammar package unavailable" in result.diagnostics[0]
    assert {directory: snapshot(directory) for directory in watched} == before


def test_unsupported_language_uses_complete_lexical_fallback() -> None:
    source = "alpha = 1\nbeta = 2\n"

    result = extract_structure("src/example.unknown", "unknown", source)

    assert result.structural_mode == "lexical"
    assert result.symbols == ()
    assert "unsupported language" in result.diagnostics[0]
    _assert_complete(source, result)


@requires_structure_pack
def test_malformed_parse_uses_lexical_fallback() -> None:
    # Presence-dependent despite asserting the lexical result: this covers the
    # *parse-failure* fallback, which requires an installed grammar to attempt
    # and fail. The missing-package fallback is covered separately by
    # test_missing_optional_package_falls_back_without_network_or_writes, which
    # runs in every environment.
    source = (FIXTURES / "malformed.py").read_text(encoding="utf-8")

    result = extract_structure("src/malformed.py", "python", source)

    assert result.structural_mode == "lexical"
    assert result.symbols == ()
    assert "lexical fallback" in result.diagnostics[0]
    _assert_complete(source, result)


def test_lexical_atoms_are_gap_free_for_unicode_and_crlf() -> None:
    source = "café = '雪'\r\nemoji = '🙂'\r\n"

    atoms = lexical_atoms("src\\unicode.py", "python", source)

    assert [atom.range.start_byte for atom in atoms] == [
        0,
        len(atoms[0].source.encode("utf-8")),
    ]
    assert (atoms[0].range.end_line, atoms[0].range.end_column) == (2, 1)
    before_newline = "café = '雪'"
    assert source_range_for_slice(
        source,
        0,
        len(before_newline.encode("utf-8")),
    ).end_column == len(before_newline.encode("utf-8")) + 1
    assert b"".join(
        atom.source.encode("utf-8") for atom in atoms
    ) == source.encode("utf-8")
    assert atoms[0].rel_path == "src/unicode.py"


def test_unicode_and_crlf_anchor_coordinates_are_stable() -> None:
    """Pin the exact anchors the coordinate index must keep producing.

    Columns are 1-based UTF-8 *byte* columns, a ``\\r\\n`` pair belongs to the
    line it terminates, and only code-point boundaries resolve.  These literals
    guard the bisect-based index against any drift from the per-character
    coordinate table it replaced.
    """
    source = "café = '雪'\r\nemoji = '🙂'\r\nplain\n"

    expected = (
        (0, 1, 1),
        (3, 1, 4),
        (5, 1, 6),
        (13, 1, 14),
        (15, 2, 1),
        (24, 2, 10),
        (28, 2, 14),
        (31, 3, 1),
        (36, 3, 6),
        (37, 4, 1),
    )
    for byte_offset, line, column in expected:
        source_range = source_range_for_slice(source, byte_offset, byte_offset)
        assert (source_range.start_line, source_range.start_column) == (line, column)
        assert (source_range.end_line, source_range.end_column) == (line, column)

    # Continuation bytes of é, 雪, and 🙂 are the only rejected offsets, and a
    # negative or past-the-end endpoint is rejected too.
    rejected = [
        byte_offset
        for byte_offset in range(-1, len(source.encode("utf-8")) + 2)
        if not _resolves(source, byte_offset)
    ]
    assert rejected == [-1, 4, 10, 11, 25, 26, 27, 38]


def _resolves(source: str, byte_offset: int) -> bool:
    try:
        source_range_for_slice(source, byte_offset, byte_offset)
    except ValueError:
        return False
    return True


def test_oversized_line_uses_codepoint_safe_continuations() -> None:
    source = "é" * 9

    atoms = lexical_atoms("big.py", "python", source, max_atom_chars=4)

    assert [atom.source for atom in atoms] == ["é" * 4, "é" * 4, "é"]
    assert [atom.range.end_byte - atom.range.start_byte for atom in atoms] == [8, 8, 2]
    assert b"".join(
        atom.source.encode("utf-8") for atom in atoms
    ) == source.encode("utf-8")


def test_structure_contract_rejects_gap_and_mismatched_domain_id() -> None:
    source = "abc"
    bad_atom = Atom(
        atom_id=atom_id_for("a.py", "line", 1, 3),
        rel_path="a.py",
        language="python",
        kind="line",
        name="line 1",
        range=SourceRange(1, 3, 1, 2, 1, 4),
        source="bc",
    )

    with pytest.raises(ValueError, match="gap-free"):
        validate_structure(source, StructureResult("syntax", (bad_atom,)))
    with pytest.raises(ValueError, match="atom_id"):
        Atom(
            atom_id="atom_" + "0" * 64,
            rel_path="a.py",
            language="python",
            kind="line",
            name="line 1",
            range=SourceRange(0, 3, 1, 1, 1, 4),
            source=source,
        )


def test_source_range_rejects_bool_and_reversed_coordinates() -> None:
    with pytest.raises(ValueError):
        SourceRange(True, 1, 1, 1, 1, 2)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SourceRange(2, 1, 1, 1, 1, 2)
    with pytest.raises(ValueError):
        SourceRange(0, 1, 2, 1, 1, 2)


def test_declaration_extraction_stops_at_configured_limit(monkeypatch) -> None:
    name_lookups = [0]

    class FakeName:
        type = "identifier"
        children = ()

        def __init__(self, start: int) -> None:
            self.start_byte = start
            self.end_byte = start + 1

    class FakeDeclaration:
        type = "function_definition"
        children = ()

        def __init__(self, start: int, parent) -> None:
            self.start_byte = start
            self.end_byte = start + 1
            self.parent = parent

        def child_by_field_name(self, field: str):
            name_lookups[0] += 1
            return FakeName(self.start_byte) if field == "name" else None

    class FakeRoot:
        type = "module"

        def __init__(self, count: int) -> None:
            self.children = tuple(
                FakeDeclaration(index, self) for index in range(count)
            )

    limit = 17
    monkeypatch.setattr(
        tree_sitter_structure,
        "MAX_EXTRACTED_DECLARATIONS",
        limit,
    )
    declarations = tree_sitter_structure._declarations(
        FakeRoot(100),
        tree_sitter_structure.SourceIndex("x" * 100),
        LANGUAGE_SPECS["python"],
    )

    assert len(declarations) == limit
    assert name_lookups == [limit]


def test_outer_declaration_selection_scales_for_many_siblings() -> None:
    declarations = [
        tree_sitter_structure._Declaration(
            kind="function_definition",
            start_byte=index * 2,
            end_byte=index * 2 + 1,
            name_start_byte=index * 2,
            name=f"f{index}",
            signature=f"def f{index}()",
        )
        for index in range(10_000)
    ]

    assert tree_sitter_structure._outer_declarations(declarations) == declarations


@requires_structure_pack
def test_many_sibling_declarations_have_stable_ownership_and_order() -> None:
    count = 4_096
    source = "".join(
        f"def function_{index}(): return {index}\n" for index in range(count)
    )

    result = extract_structure("src/many.py", "python", source)

    assert result.structural_mode == "syntax", result.diagnostics
    _assert_complete(source, result)
    assert len(result.atoms) == count
    assert len(result.symbols) == count
    assert [symbol.qualified_name for symbol in result.symbols] == [
        f"function_{index}" for index in range(count)
    ]
    assert all(
        atom.symbol_ids == (symbol.symbol_id,)
        for atom, symbol in zip(result.atoms, result.symbols)
    )
