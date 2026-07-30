"""Read-only optional tree-sitter extraction with complete lexical fallback."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Iterable

from codedoc.parser.language_specs import LanguageSpec, spec_for
from codedoc.parser.source_structure import (
    MAX_STRUCTURE_DIAGNOSTIC_CHARS,
    MAX_STRUCTURE_NAME_CHARS,
    MAX_STRUCTURE_SIGNATURE_CHARS,
    Atom,
    SourceRange,
    StructureResult,
    SymbolFact,
    atom_id_for,
    normalize_rel_path,
    symbol_id_for,
)


def _parser_package_version() -> str:
    """Return installed bundle identity without importing the optional parser."""
    try:
        return version("tree-sitter-language-pack")
    except PackageNotFoundError:
        return "not-installed"


PARSER_PACKAGE_VERSION = _parser_package_version()
MAX_STRUCTURE_DIAGNOSTICS = 32
MAX_EXTRACTED_DECLARATIONS = 8192 + 1
_NAME_NODE_TYPES = frozenset(
    {
        "identifier",
        "type_identifier",
        "property_identifier",
        "field_identifier",
        "simple_identifier",
        "constant",
    }
)


def _line_start_offsets(data: bytes) -> list[int]:
    """Return the byte offset of every line start in *data*.

    A ``\\n`` always begins a new line; a ``\\r`` begins one only when it is not
    the first half of a ``\\r\\n`` pair.  ``0x0A``/``0x0D`` can never appear
    inside a multi-byte UTF-8 sequence, so scanning bytes is exactly equivalent
    to the character-by-character scan this replaces, without encoding one
    character at a time.
    """
    starts = [0]
    index = data.find(b"\n")
    while index != -1:
        starts.append(index + 1)
        index = data.find(b"\n", index + 1)
    index = data.find(b"\r")
    while index != -1:
        if index + 1 >= len(data) or data[index + 1] != 0x0A:
            starts.append(index + 1)
        index = data.find(b"\r", index + 1)
    starts.sort()
    return starts


class SourceIndex:
    """Coordinate and UTF-8 boundary index over one canonical decoded snapshot.

    Building one index costs O(len(content)) and stores only the line-start byte
    offsets; every anchor lookup is an O(log lines) :func:`bisect_right` plus
    arithmetic, and a code-point boundary is recognized from the single byte at
    the endpoint.  Nothing is retained per source character.

    Callers that resolve many ranges over the same snapshot must build one index
    and reuse it — see :func:`codedoc.core.file_division.pack_chunks`.  Columns
    are 1-based UTF-8 *byte* columns within their line, and every anchor,
    boundary rejection, and line lookup is byte-identical to the per-character
    coordinate table this replaces.
    """

    def __init__(self, content: str) -> None:
        if not isinstance(content, str):
            raise ValueError("source content must be text.")
        self.content = content
        self.data = content.encode("utf-8")
        self._line_starts: list[int] = _line_start_offsets(self.data)

    def _anchor(self, byte_offset: int) -> tuple[int, int]:
        """Return the 1-based line and UTF-8 byte column at one boundary offset.

        A negative, past-the-end, non-integer, or continuation-byte endpoint is
        rejected exactly where the previous coordinate lookup raised a
        ``KeyError``.
        """
        data = self.data
        if (
            not isinstance(byte_offset, int)
            or isinstance(byte_offset, bool)
            or byte_offset < 0
            or byte_offset > len(data)
            or (byte_offset < len(data) and (data[byte_offset] & 0xC0) == 0x80)
        ):
            raise ValueError("range endpoint is not a UTF-8 code-point boundary.")
        line = bisect_right(self._line_starts, byte_offset)
        return line, byte_offset - self._line_starts[line - 1] + 1

    def range(self, start_byte: int, end_byte: int) -> SourceRange:
        start_line, start_column = self._anchor(start_byte)
        end_line, end_column = self._anchor(end_byte)
        if end_byte > len(self.data):
            raise ValueError("range endpoint is outside the source snapshot.")
        return SourceRange(
            start_byte,
            end_byte,
            start_line,
            start_column,
            end_line,
            end_column,
        )

    def slice(self, start_byte: int, end_byte: int) -> str:
        self.range(start_byte, end_byte)
        return self.data[start_byte:end_byte].decode("utf-8")

    def line_start(self, byte_offset: int) -> int:
        self.range(byte_offset, byte_offset)
        return self._line_starts[bisect_right(self._line_starts, byte_offset) - 1]

    def line_end(self, byte_offset: int) -> int:
        current_start = self.line_start(byte_offset)
        if byte_offset == current_start and byte_offset > 0:
            return byte_offset
        following = bisect_right(self._line_starts, current_start)
        if following < len(self._line_starts):
            return self._line_starts[following]
        return len(self.data)


def source_range_for_slice(
    content: str, start_byte: int, end_byte: int
) -> SourceRange:
    """Return canonical anchors for one UTF-8 byte slice.

    Convenience wrapper for one-off callers; resolving many ranges over one
    snapshot must reuse a single :class:`SourceIndex` instead.
    """
    return SourceIndex(content).range(start_byte, end_byte)


@dataclass(frozen=True, slots=True)
class _Declaration:
    kind: str
    start_byte: int
    end_byte: int
    name_start_byte: int
    name: str
    signature: str
    embedded: bool = False


@dataclass(frozen=True, slots=True)
class _EmbeddedRegion:
    start_byte: int
    end_byte: int


def _bounded_diagnostic(prefix: str, exc: object) -> str:
    message = " ".join(str(exc).split())
    return f"{prefix}: {message}"[:MAX_STRUCTURE_DIAGNOSTIC_CHARS]


def _physical_lines(content: str) -> tuple[str, ...]:
    lines = tuple(content.splitlines(keepends=True))
    if lines:
        return lines
    return (content,) if content else ()


def lexical_atoms(
    rel_path: str,
    language: str,
    content: str,
    *,
    max_atom_chars: int | None = None,
) -> tuple[Atom, ...]:
    """Return complete physical-line atoms, splitting only an oversized line."""
    rel = normalize_rel_path(rel_path)
    source_index = SourceIndex(content)
    atoms: list[Atom] = []
    offset = 0
    for line_number, line in enumerate(_physical_lines(content), start=1):
        pieces = _continuation_pieces(line, max_atom_chars)
        for piece_index, piece in enumerate(pieces):
            start = offset
            end = start + len(piece.encode("utf-8"))
            continuation = len(pieces) > 1
            kind = "line-continuation" if continuation else "line"
            name = (
                f"line {line_number}.{piece_index + 1}"
                if continuation
                else f"line {line_number}"
            )
            source_range = source_index.range(start, end)
            atoms.append(
                Atom(
                    atom_id=atom_id_for(rel, kind, start, end),
                    rel_path=rel,
                    language=language,
                    kind=kind,
                    name=name,
                    range=source_range,
                    source=piece,
                    symbol_ids=(),
                )
            )
            offset = end
    if offset != len(source_index.data):
        raise ValueError("lexical atoms did not consume the complete source.")
    return tuple(atoms)


def _continuation_pieces(line: str, max_atom_chars: int | None) -> tuple[str, ...]:
    if max_atom_chars is None:
        return (line,)
    if isinstance(max_atom_chars, bool) or not isinstance(max_atom_chars, int):
        raise ValueError("max_atom_chars must be an integer or None.")
    if max_atom_chars <= 0:
        raise ValueError("max_atom_chars must be greater than zero.")
    if len(line) <= max_atom_chars:
        return (line,)
    return tuple(
        line[index : index + max_atom_chars]
        for index in range(0, len(line), max_atom_chars)
    )


def extract_structure(
    rel_path: str,
    language: str,
    content: str,
    *,
    max_atom_chars: int | None = None,
) -> StructureResult:
    """Extract syntax facts or a complete lexical fallback without side effects."""
    rel = normalize_rel_path(rel_path)
    spec = spec_for(language)
    if spec is None:
        return _lexical_result(
            rel,
            language,
            content,
            max_atom_chars,
            f"unsupported language for structural extraction: {language}",
        )
    try:
        from tree_sitter_language_pack import get_parser  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:
        return _lexical_result(
            rel,
            language,
            content,
            max_atom_chars,
            _bounded_diagnostic("optional bundled grammar package unavailable", exc),
        )

    try:
        if language == "html":
            result = _extract_html(rel, language, content, spec, get_parser)
        else:
            parser = get_parser(spec.grammar_name)
            tree = parser.parse(content.encode("utf-8"))
            if not _parse_is_usable(tree.root_node, len(content.encode("utf-8"))):
                raise ValueError("tree-sitter parse is error-dominated")
            result = _syntax_result(rel, language, content, tree.root_node, spec)
        validate_structure(content, result)
        return result
    except Exception as exc:
        return _lexical_result(
            rel,
            language,
            content,
            max_atom_chars,
            _bounded_diagnostic("lexical fallback", exc),
        )


def _lexical_result(
    rel_path: str,
    language: str,
    content: str,
    max_atom_chars: int | None,
    diagnostic: str,
) -> StructureResult:
    result = StructureResult(
        "lexical",
        lexical_atoms(
            rel_path,
            language,
            content,
            max_atom_chars=max_atom_chars,
        ),
        (),
        (diagnostic[:MAX_STRUCTURE_DIAGNOSTIC_CHARS],),
    )
    validate_structure(content, result)
    return result


def _walk(node) -> Iterable:
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(getattr(current, "children", ())))


def _parse_is_usable(root, source_bytes: int) -> bool:
    if root is None:
        return False
    if not getattr(root, "has_error", False):
        return True
    named_count = 0
    error_count = 0
    error_bytes = 0
    for node in _walk(root):
        if getattr(node, "is_named", False):
            named_count += 1
        if (
            getattr(node, "is_error", False)
            or getattr(node, "is_missing", False)
            or getattr(node, "type", "") == "ERROR"
        ):
            error_count += 1
            error_bytes += max(0, int(node.end_byte) - int(node.start_byte))
    if error_count == 0:
        return False
    return not (
        error_count >= 4
        or error_count * 5 >= max(1, named_count)
        or error_bytes * 5 >= max(1, source_bytes)
    )


def _node_name(node, source_index: SourceIndex, spec: LanguageSpec) -> tuple[str, int]:
    for field in spec.name_fields:
        name_node = node.child_by_field_name(field)
        if name_node is not None:
            text = source_index.slice(name_node.start_byte, name_node.end_byte).strip()
            if text:
                return text[:MAX_STRUCTURE_NAME_CHARS], int(name_node.start_byte)
    stack = list(reversed(getattr(node, "children", ())))
    while stack:
        descendant = stack.pop()
        descendant_type = getattr(descendant, "type", "")
        if descendant_type in _NAME_NODE_TYPES:
            text = source_index.slice(
                descendant.start_byte, descendant.end_byte
            ).strip()
            if text:
                return (
                    text[:MAX_STRUCTURE_NAME_CHARS],
                    int(descendant.start_byte),
                )
        if descendant_type not in spec.declaration_node_types:
            stack.extend(reversed(getattr(descendant, "children", ())))
    return "", int(node.start_byte)


def _declaration_span_node(node, spec: LanguageSpec):
    parent = getattr(node, "parent", None)
    if (
        spec.language == "python"
        and parent is not None
        and getattr(parent, "type", "") == "decorated_definition"
    ):
        return parent
    return node


def _signature_for(node, source_index: SourceIndex, name: str) -> str:
    start = int(node.start_byte)
    end = min(
        int(node.end_byte),
        start + MAX_STRUCTURE_SIGNATURE_CHARS * 4,
    )
    while (
        start < end < len(source_index.data)
        and (source_index.data[end] & 0xC0) == 0x80
    ):
        end -= 1
    first_line = source_index.slice(start, end).splitlines()
    return (first_line[0] if first_line else name)[:MAX_STRUCTURE_SIGNATURE_CHARS]


def _declarations(
    root,
    source_index: SourceIndex,
    spec: LanguageSpec,
    *,
    byte_offset: int = 0,
    embedded: bool = False,
    limit: int | None = None,
) -> list[_Declaration]:
    maximum = MAX_EXTRACTED_DECLARATIONS if limit is None else max(0, limit)
    if maximum == 0:
        return []
    declarations: list[_Declaration] = []
    for node in _walk(root):
        if getattr(node, "type", "") not in spec.declaration_node_types:
            continue
        name, name_start = _node_name(node, source_index, spec)
        if not name:
            continue
        span_node = _declaration_span_node(node, spec)
        start = byte_offset + int(span_node.start_byte)
        end = byte_offset + int(span_node.end_byte)
        absolute_name_start = byte_offset + name_start
        declarations.append(
            _Declaration(
                kind=str(node.type),
                start_byte=start,
                end_byte=end,
                name_start_byte=absolute_name_start,
                name=name,
                signature=_signature_for(node, source_index, name),
                embedded=embedded,
            )
        )
        if len(declarations) >= maximum:
            break
    return declarations


def _outer_declarations(
    declarations: list[_Declaration],
) -> list[_Declaration]:
    selected: list[_Declaration] = []
    for declaration in sorted(
        declarations, key=lambda item: (item.start_byte, -item.end_byte, item.kind)
    ):
        if selected and declaration.start_byte < selected[-1].end_byte:
            continue
        selected.append(declaration)
    return selected


def _containing_region(
    start: int,
    end: int,
    regions: tuple[_EmbeddedRegion, ...],
    region_starts: list[int],
) -> _EmbeddedRegion | None:
    index = bisect_right(region_starts, start) - 1
    if index < 0:
        return None
    region = regions[index]
    if region.start_byte <= start and end <= region.end_byte:
        return region
    return None


def _overlapping_declaration(
    start: int,
    end: int,
    declarations: list[_Declaration],
    declaration_starts: list[int],
) -> _Declaration | None:
    position = bisect_right(declaration_starts, start)
    if position:
        candidate = declarations[position - 1]
        if candidate.start_byte < end and start < candidate.end_byte:
            return candidate
    if position < len(declarations):
        candidate = declarations[position]
        if candidate.start_byte < end and start < candidate.end_byte:
            return candidate
    return None


def _syntax_result(
    rel_path: str,
    language: str,
    content: str,
    root,
    spec: LanguageSpec,
    *,
    precomputed_declarations: tuple[_Declaration, ...] | None = None,
    embedded_regions: tuple[_EmbeddedRegion, ...] = (),
) -> StructureResult:
    source_index = SourceIndex(content)
    declarations = list(
        _declarations(root, source_index, spec)
        if precomputed_declarations is None
        else precomputed_declarations[:MAX_EXTRACTED_DECLARATIONS]
    )
    ordered_regions = tuple(
        sorted(
            embedded_regions,
            key=lambda region: (region.start_byte, region.end_byte),
        )
    )
    region_starts = [region.start_byte for region in ordered_regions]
    previous_region_end = 0
    for region in ordered_regions:
        if not (0 <= region.start_byte < region.end_byte <= len(source_index.data)):
            raise ValueError("embedded range is outside the host source.")
        if region.start_byte < previous_region_end:
            raise ValueError("embedded ranges overlap.")
        previous_region_end = region.end_byte
    if any(
        declaration.embedded
        and _containing_region(
            declaration.start_byte,
            declaration.end_byte,
            ordered_regions,
            region_starts,
        )
        is None
        for declaration in declarations
    ):
        raise ValueError("embedded declaration is outside an embedded range.")
    boundaries = {0, len(source_index.data)}
    for region in ordered_regions:
        boundaries.add(region.start_byte)
        boundaries.add(region.end_byte)
    host_declarations = [
        declaration for declaration in declarations if not declaration.embedded
    ]
    embedded_declarations = [
        declaration for declaration in declarations if declaration.embedded
    ]
    host_owners = _outer_declarations(host_declarations)
    embedded_owners = _outer_declarations(embedded_declarations)
    host_owner_starts = [declaration.start_byte for declaration in host_owners]
    embedded_owner_starts = [declaration.start_byte for declaration in embedded_owners]
    boundary_declarations = [*host_owners, *embedded_declarations]
    for declaration in boundary_declarations:
        if declaration.embedded:
            boundaries.add(declaration.start_byte)
            boundaries.add(declaration.end_byte)
        else:
            boundaries.add(source_index.line_start(declaration.start_byte))
            boundaries.add(source_index.line_end(declaration.end_byte))
    ordered = sorted(boundaries)
    atoms: list[Atom] = []
    for index, (start, end) in enumerate(zip(ordered, ordered[1:]), start=1):
        if end <= start:
            continue
        embedded_region = _containing_region(
            start,
            end,
            ordered_regions,
            region_starts,
        )
        owner_declarations = (
            embedded_owners if embedded_region is not None else host_owners
        )
        owner_starts = (
            embedded_owner_starts if embedded_region is not None else host_owner_starts
        )
        owner = _overlapping_declaration(
            start,
            end,
            owner_declarations,
            owner_starts,
        )
        if owner is not None:
            kind = owner.kind
            name = owner.name
        elif embedded_region is not None:
            kind = "embedded_javascript_gap"
            name = f"inline JavaScript segment {index}"
        else:
            kind = "syntax-gap"
            name = f"source segment {index}"
        atoms.append(
            Atom(
                atom_id=atom_id_for(rel_path, kind, start, end),
                rel_path=rel_path,
                language=language,
                kind=kind,
                name=name,
                range=source_index.range(start, end),
                source=source_index.slice(start, end),
                symbol_ids=(),
            )
        )

    symbols: list[SymbolFact] = []
    atom_starts = [atom.range.start_byte for atom in atoms]
    for declaration in declarations:
        owner_index = bisect_right(atom_starts, declaration.name_start_byte) - 1
        if owner_index < 0:
            continue
        owner = atoms[owner_index]
        if declaration.name_start_byte >= owner.range.end_byte:
            continue
        source_range = source_index.range(declaration.start_byte, declaration.end_byte)
        symbols.append(
            SymbolFact(
                symbol_id=symbol_id_for(
                    rel_path,
                    declaration.kind,
                    declaration.name,
                    declaration.start_byte,
                    declaration.end_byte,
                ),
                rel_path=rel_path,
                language=language,
                kind=declaration.kind,
                qualified_name=declaration.name,
                signature=declaration.signature,
                range=source_range,
                atom_id=owner.atom_id,
            )
        )

    symbol_ids_by_atom: dict[str, list[str]] = {}
    for symbol in symbols:
        symbol_ids_by_atom.setdefault(symbol.atom_id, []).append(symbol.symbol_id)
    atoms_with_symbols = tuple(
        Atom(
            atom.atom_id,
            atom.rel_path,
            atom.language,
            atom.kind,
            atom.name,
            atom.range,
            atom.source,
            tuple(symbol_ids_by_atom.get(atom.atom_id, ())),
        )
        for atom in atoms
    )
    return StructureResult("syntax", atoms_with_symbols, tuple(symbols), ())


def _extract_html(rel_path: str, language: str, content: str, spec, get_parser):
    """Parse HTML plus every inline JavaScript body with canonical remapping."""
    source_index = SourceIndex(content)
    host_tree = get_parser(spec.grammar_name).parse(source_index.data)
    if not _parse_is_usable(host_tree.root_node, len(source_index.data)):
        raise ValueError("HTML parse is error-dominated")

    javascript_parser = None
    javascript_spec = spec_for("javascript")
    declarations = _declarations(
        host_tree.root_node,
        source_index,
        spec,
    )
    embedded: list[_Declaration] = []
    embedded_regions: list[_EmbeddedRegion] = []
    if javascript_spec is None:
        raise ValueError("JavaScript structural specification is unavailable")
    for node in _walk(host_tree.root_node):
        if getattr(node, "type", "") != "script_element":
            continue
        raw_nodes = [
            child
            for child in getattr(node, "children", ())
            if getattr(child, "type", "") in {"raw_text", "script_text"}
        ]
        for raw_node in raw_nodes:
            raw_start = int(raw_node.start_byte)
            raw_end = int(raw_node.end_byte)
            raw = source_index.data[raw_start:raw_end]
            if raw:
                embedded_regions.append(_EmbeddedRegion(raw_start, raw_end))
            if not raw.strip():
                continue
            if javascript_parser is None:
                javascript_parser = get_parser("javascript")
            tree = javascript_parser.parse(raw)
            if not _parse_is_usable(tree.root_node, len(raw)):
                raise ValueError("inline JavaScript parse is error-dominated")
            remaining = MAX_EXTRACTED_DECLARATIONS - len(declarations) - len(embedded)
            if remaining > 0:
                embedded_index = SourceIndex(raw.decode("utf-8"))
                embedded.extend(
                    _declarations(
                        tree.root_node,
                        embedded_index,
                        javascript_spec,
                        byte_offset=raw_start,
                        embedded=True,
                        limit=remaining,
                    )
                )

    return _syntax_result(
        rel_path,
        language,
        content,
        host_tree.root_node,
        spec,
        precomputed_declarations=tuple([*declarations, *embedded]),
        embedded_regions=tuple(embedded_regions),
    )


def validate_structure(content: str, result: StructureResult) -> None:
    """Validate all snapshot-dependent structure invariants."""
    source_index = SourceIndex(content)
    offset = 0
    for atom in result.atoms:
        source_range = atom.range
        if source_range.start_byte != offset:
            raise ValueError("atoms are not an ordered gap-free partition.")
        if source_range.end_byte <= source_range.start_byte:
            raise ValueError("atoms must own non-empty ranges.")
        if source_range != source_index.range(
            source_range.start_byte, source_range.end_byte
        ):
            raise ValueError("atom line/column anchors do not map to byte endpoints.")
        if atom.source != source_index.slice(
            source_range.start_byte, source_range.end_byte
        ):
            raise ValueError("atom source does not equal its canonical decoded slice.")
        offset = source_range.end_byte
    if offset != len(source_index.data):
        raise ValueError("atoms do not cover the complete canonical source.")
    if content and not result.atoms:
        raise ValueError("non-empty source must contain at least one atom.")
    for symbol in result.symbols:
        if symbol.range != source_index.range(
            symbol.range.start_byte, symbol.range.end_byte
        ):
            raise ValueError("symbol anchors do not map to byte endpoints.")
        if not (
            0 <= symbol.range.start_byte < symbol.range.end_byte <= len(source_index.data)
        ):
            raise ValueError("symbol range is outside the canonical source.")
