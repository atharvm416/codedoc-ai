"""Immutable source-structure values for deterministic file division."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

_ID_SEP = "\x1f"
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_STRUCTURE_DIAGNOSTICS = 32
MAX_STRUCTURE_DIAGNOSTIC_CHARS = 400
MAX_STRUCTURE_NAME_CHARS = 240
MAX_STRUCTURE_KIND_CHARS = 120
MAX_STRUCTURE_SIGNATURE_CHARS = 600


def normalize_rel_path(rel_path: str) -> str:
    """Return one canonical project-relative POSIX path."""
    if not isinstance(rel_path, str) or "\x00" in rel_path:
        raise ValueError("rel_path must be text without NUL characters.")
    raw = rel_path.replace("\\", "/")
    candidate = PurePosixPath(raw)
    if (
        not raw
        or candidate.is_absolute()
        or ".." in candidate.parts
        or raw.startswith("/")
    ):
        raise ValueError(f"rel_path must be a project-relative path: {rel_path!r}")
    normalized = candidate.as_posix()
    if normalized in ("", "."):
        raise ValueError(f"rel_path must name a file: {rel_path!r}")
    return normalized


def _real_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def _bounded_text(
    value: object,
    name: str,
    *,
    maximum: int,
    required: bool = True,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text.")
    if required and not value:
        raise ValueError(f"{name} is required.")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters.")
    return value


def _digest_id(prefix: str, *parts: object) -> str:
    payload = _ID_SEP.join((prefix, *(str(part) for part in parts)))
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def atom_id_for(rel_path: str, kind: str, start_byte: int, end_byte: int) -> str:
    return _digest_id(
        "atom",
        normalize_rel_path(rel_path),
        kind,
        _real_int(start_byte, "start_byte"),
        _real_int(end_byte, "end_byte"),
    )


def symbol_id_for(
    rel_path: str,
    kind: str,
    name: str,
    start_byte: int,
    end_byte: int,
) -> str:
    return _digest_id(
        "symbol",
        normalize_rel_path(rel_path),
        kind,
        name,
        _real_int(start_byte, "start_byte"),
        _real_int(end_byte, "end_byte"),
    )


def _validate_domain_id(value: object, prefix: str, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text.")
    expected_prefix = f"{prefix}_"
    if not value.startswith(expected_prefix) or not _HEX_64_RE.fullmatch(
        value[len(expected_prefix) :]
    ):
        raise ValueError(f"{name} must be a domain-separated SHA-256 ID.")
    return value


@dataclass(frozen=True, slots=True)
class SourceRange:
    """One half-open UTF-8 byte range with 1-based end-exclusive anchors."""

    start_byte: int
    end_byte: int
    start_line: int
    start_column: int
    end_line: int
    end_column: int

    def __post_init__(self) -> None:
        for name in ("start_byte", "end_byte"):
            _real_int(getattr(self, name), name)
        for name in ("start_line", "start_column", "end_line", "end_column"):
            _real_int(getattr(self, name), name, minimum=1)
        if self.end_byte < self.start_byte:
            raise ValueError("source range bytes must be non-reversed.")
        if (self.end_line, self.end_column) < (self.start_line, self.start_column):
            raise ValueError("source range coordinates must be non-reversed.")

    def to_public(self) -> dict:
        return {
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "start": {"line": self.start_line, "column": self.start_column},
            "end": {"line": self.end_line, "column": self.end_column},
        }


@dataclass(frozen=True, slots=True)
class Atom:
    atom_id: str
    rel_path: str
    language: str
    kind: str
    name: str
    range: SourceRange
    source: str
    symbol_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        rel_path = normalize_rel_path(self.rel_path)
        language = _bounded_text(
            self.language, "atom language", maximum=64
        )
        kind = _bounded_text(self.kind, "atom kind", maximum=MAX_STRUCTURE_KIND_CHARS)
        _bounded_text(
            self.name,
            "atom name",
            maximum=MAX_STRUCTURE_NAME_CHARS,
            required=False,
        )
        if not isinstance(self.range, SourceRange):
            raise ValueError("atom range must be a SourceRange.")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("atom source must be non-empty text.")
        if len(self.source.encode("utf-8")) != self.range.end_byte - self.range.start_byte:
            raise ValueError("atom source byte length does not match its range.")
        symbol_ids = tuple(self.symbol_ids)
        for symbol_id in symbol_ids:
            _validate_domain_id(symbol_id, "symbol", "atom symbol_id")
        if len(symbol_ids) != len(set(symbol_ids)):
            raise ValueError("atom has duplicate symbol IDs.")
        expected_id = atom_id_for(
            rel_path, kind, self.range.start_byte, self.range.end_byte
        )
        if self.atom_id != expected_id:
            raise ValueError("atom_id does not match the atom path/range/kind.")
        object.__setattr__(self, "rel_path", rel_path)
        object.__setattr__(self, "language", language)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "symbol_ids", symbol_ids)


@dataclass(frozen=True, slots=True)
class SymbolFact:
    symbol_id: str
    rel_path: str
    language: str
    kind: str
    qualified_name: str
    signature: str
    range: SourceRange
    atom_id: str

    def __post_init__(self) -> None:
        rel_path = normalize_rel_path(self.rel_path)
        language = _bounded_text(
            self.language, "symbol language", maximum=64
        )
        kind = _bounded_text(
            self.kind, "symbol kind", maximum=MAX_STRUCTURE_KIND_CHARS
        )
        name = _bounded_text(
            self.qualified_name,
            "symbol qualified_name",
            maximum=MAX_STRUCTURE_NAME_CHARS,
        )
        _bounded_text(
            self.signature,
            "symbol signature",
            maximum=MAX_STRUCTURE_SIGNATURE_CHARS,
            required=False,
        )
        if not isinstance(self.range, SourceRange):
            raise ValueError("symbol range must be a SourceRange.")
        _validate_domain_id(self.atom_id, "atom", "symbol atom_id")
        expected_id = symbol_id_for(
            rel_path, kind, name, self.range.start_byte, self.range.end_byte
        )
        if self.symbol_id != expected_id:
            raise ValueError("symbol_id does not match the symbol path/range/kind/name.")
        object.__setattr__(self, "rel_path", rel_path)
        object.__setattr__(self, "language", language)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "qualified_name", name)


@dataclass(frozen=True, slots=True)
class StructureResult:
    structural_mode: Literal["syntax", "lexical"]
    atoms: tuple[Atom, ...]
    symbols: tuple[SymbolFact, ...] = ()
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.structural_mode not in ("syntax", "lexical"):
            raise ValueError("unsupported structural mode.")
        atoms = tuple(self.atoms)
        symbols = tuple(self.symbols)
        diagnostics = tuple(self.diagnostics)
        if len(diagnostics) > MAX_STRUCTURE_DIAGNOSTICS:
            raise ValueError("too many structure diagnostics.")
        for diagnostic in diagnostics:
            _bounded_text(
                diagnostic,
                "structure diagnostic",
                maximum=MAX_STRUCTURE_DIAGNOSTIC_CHARS,
            )
        atom_ids = [atom.atom_id for atom in atoms]
        symbol_ids = [symbol.symbol_id for symbol in symbols]
        if len(atom_ids) != len(set(atom_ids)):
            raise ValueError("duplicate atom IDs.")
        if len(symbol_ids) != len(set(symbol_ids)):
            raise ValueError("duplicate symbol IDs.")
        atom_by_id = {atom.atom_id: atom for atom in atoms}
        symbol_ids_by_atom: dict[str, list[str]] = {}
        for symbol in symbols:
            owner = atom_by_id.get(symbol.atom_id)
            if owner is None:
                raise ValueError("symbol references an unknown atom.")
            if owner.rel_path != symbol.rel_path or owner.language != symbol.language:
                raise ValueError("symbol and owning atom path/language disagree.")
            if not (
                owner.range.start_byte
                <= symbol.range.start_byte
                < owner.range.end_byte
            ):
                raise ValueError("symbol declaration start is outside its owning atom.")
            symbol_ids_by_atom.setdefault(symbol.atom_id, []).append(symbol.symbol_id)
        for atom in atoms:
            if atom.symbol_ids != tuple(symbol_ids_by_atom.get(atom.atom_id, ())):
                raise ValueError("atom symbol_ids do not match anchored symbols.")
        if self.structural_mode == "lexical" and symbols:
            raise ValueError("lexical structure must not fabricate symbols.")
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "diagnostics", diagnostics)
