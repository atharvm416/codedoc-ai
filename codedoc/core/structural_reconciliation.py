"""One shared source-backed authority for the public ``functions`` /
``classes`` / ``exports`` arrays.

Every publication route -- single and triple ordinary, single and triple
truncate, split final synthesis, and recovery re-assembly -- delegates here so
there is exactly one definition of a "real function", "real class", and "real
export". A model response may only *describe* a declaration the source proves;
it can never add, duplicate, or reclassify one.

Two evidence sources back a declaration, in priority order:

* parser ``SymbolFact`` values (syntax structural mode). These own declaration
  identity, kind, qualified name, signature, and source order.
* a deliberately incomplete, language-specific lexical recognizer over the
  canonical visible source. It supplements constructs a grammar does not model
  as declarations (arrow/function-expression bindings, typed React components)
  and is the sole authority when no symbols exist (base install).

When neither evidence source can speak for a declaration -- no parser symbol
and no conservative lexical proof for the file's language -- the model's item
is omitted. There is no passthrough: a structural fact is never published
without syntax or conservative lexical proof, for any language.

Two parser kinds and a name are never enough to guess an overload apart: when
the model supplies a signature and the source proves several same-name
declarations, the description attaches only to the one whose authoritative
signature is compatible; otherwise every proven same-name declaration is
published without a guessed description. Response order and split-leaf order
never decide final source order.

Diagnostics expose only counts, reason codes, and the normalized path. No
prompt, provider response, source text, signature, symbol id, or source range
ever leaves this module.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from codedoc.parser.source_structure import SymbolFact, normalize_rel_path
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

# Languages whose identifiers compare case-insensitively -- mirrors
# ``codedoc.core.file_division._CASE_INSENSITIVE_LANGUAGES`` so the split ledger
# and this authority normalize a name the same way.
_CASE_INSENSITIVE_LANGUAGES = frozenset({"sql", "pascal", "basic", "fortran"})

# Closed parser-kind -> public-bucket mapping. The set of ``kind`` strings that
# can reach here is exactly the union of ``declaration_node_types`` across
# ``codedoc.parser.language_specs.LANGUAGE_SPECS`` (the extractor only emits a
# ``SymbolFact`` for a node whose type is in its language's spec). Every entry
# below is one of those node types; anything not listed -- a container, a type
# alias, a namespace, an HTML element, an ``impl`` block, or an unknown future
# kind -- resolves to *no* publishable bucket rather than being guessed.
_FUNCTION_PARSER_KINDS = frozenset(
    {
        "function_definition",
        "function_declaration",
        "function_item",
        "function_signature",
        "generator_function_declaration",
        "arrow_function",
        "method_definition",
        "method_declaration",
        "method_signature",
        "method",
        "singleton_method",
        # A constructor is a callable member, decided explicitly here rather
        # than falling to "class" because the string contains "struct".
        "constructor_declaration",
    }
)
_CLASS_PARSER_KINDS = frozenset(
    {
        "class_definition",
        "class_declaration",
        "class_specifier",
        "class",
        "record_declaration",
        "interface_declaration",
        "protocol_declaration",
        "trait_item",
        "enum_declaration",
        "enum_item",
        "enum_specifier",
        "struct_declaration",
        "struct_item",
        "struct_specifier",
        "object_declaration",
    }
)

# Languages the conservative lexical recognizer supports today (plan section
# 5.4's initially-required set). A file in any other language with no symbols
# has no function/class/export proof path, so every model item is omitted.
_LEXICAL_PROOF_LANGUAGES = frozenset(
    {"python", "javascript", "jsx", "typescript", "tsx"}
)

_MAX_DIAGNOSTIC_COUNT = 10_000


def normalize_declaration_name(name: str, language: str) -> str:
    """Canonical comparison key for a declaration name."""
    if not isinstance(name, str):
        return ""
    normalized = " ".join(unicodedata.normalize("NFC", name).strip().split())
    if language.lower() in _CASE_INSENSITIVE_LANGUAGES:
        normalized = normalized.lower()
    return normalized


def _short_name(name: str, language: str) -> str:
    parts = re.split(r"(?:::|[./#$\\])", name)
    tail = parts[-1] if parts else name
    return normalize_declaration_name(tail, language)


def parser_kind_bucket(kind: str) -> str | None:
    """Map one parser declaration kind to its only allowed public bucket.

    Returns ``"functions"`` for a callable/component kind, ``"classes"`` for a
    class-like type kind, and ``None`` for every container, alias, namespace,
    markup element, ``impl`` block, and unrecognized kind. ``None`` means the
    symbol proves no publishable function or class -- it is omitted, never
    guessed into a bucket.
    """
    key = (kind or "").strip().lower()
    if key in _FUNCTION_PARSER_KINDS:
        return "functions"
    if key in _CLASS_PARSER_KINDS:
        return "classes"
    return None


# ---------------------------------------------------------------------------
# Conservative lexical recognizer
# ---------------------------------------------------------------------------

_IDENT = r"[A-Za-z_$À-￿][\w$À-￿]*"

# Python: a declaration keyword must open the (indentation-stripped) line.
_PY_DEF = re.compile(rf"^(?:async\s+)?def\s+({_IDENT})\b")
_PY_CLASS = re.compile(rf"^class\s+({_IDENT})\b")

# JS/TS function and class declarations, with an optional leading `export`
# (and optional `default`).
_JS_EXPORT_PREFIX = r"(?:export\s+(?:default\s+)?)?"
_JS_FUNC_DECL = re.compile(
    rf"^{_JS_EXPORT_PREFIX}(?:async\s+)?function\s*\*?\s*({_IDENT})\b"
)
_JS_CLASS_DECL = re.compile(
    rf"^{_JS_EXPORT_PREFIX}(?:abstract\s+)?class\s+({_IDENT})\b"
)

# `const`/`let`/`var NAME [: TYPE] = ...` -- the initializer decides the kind.
_JS_BINDING = re.compile(
    rf"^{_JS_EXPORT_PREFIX}(?:const|let|var)\s+({_IDENT})\s*"
    rf"(?::\s*(?P<annotation>[^=]+?)\s*)?=\s*(?P<init>.+)$"
)
# An arrow-function initializer: `(...) =>` or `IDENT =>`, optionally `async`.
_JS_ARROW_INIT = re.compile(
    rf"^(?:async\s+)?(?:\([^)]*\)|{_IDENT})\s*(?::[^=]+)?=>"
)
# A function-expression initializer: `function ...` / `async function ...`.
_JS_FUNC_INIT = re.compile(r"^(?:async\s+)?function\b")
# A typed React component annotation: `React.FC`, `FC`, `React.FunctionComponent`.
_REACT_COMPONENT_ANNOTATION = re.compile(
    r"^(?:React\.)?(?:FC|FunctionComponent|VFC|VoidFunctionComponent)\b"
)

# Explicit named re-exports: `export { A, B as C } ...`.
_JS_NAMED_EXPORT_BLOCK = re.compile(r"^export\s+(?:type\s+)?\{([^}]*)\}")
_JS_EXPORT_DEFAULT_NAME = re.compile(rf"^export\s+default\s+({_IDENT})\s*;?\s*$")
# `export * as ns from '...'` -- names the namespace binding. Bare `export *`
# names nothing and is deliberately not matched.
_JS_EXPORT_STAR_AS = re.compile(rf"^export\s+\*\s+as\s+({_IDENT})\s+from\b")

# CommonJS: an explicit single-name property export or a whole-object export.
# ``exports.NAME =`` / ``module.exports.NAME =`` bind exactly one name; a
# computed ``exports[expr] =`` is not matched. ``module.exports = NAME`` re-
# exports one local binding. ``module.exports = { ... }`` on one line lists
# shorthand or ``key:`` members.
_CJS_PROP_EXPORT = re.compile(
    rf"^(?:module\.)?exports\.({_IDENT})\s*="
)
_CJS_WHOLE_NAME = re.compile(rf"^module\.exports\s*=\s*({_IDENT})\s*;?\s*$")
_CJS_WHOLE_OBJECT = re.compile(r"^module\.exports\s*=\s*\{(.*)\}\s*;?\s*$")
_CJS_OBJECT_MEMBER = re.compile(rf"({_IDENT})\s*:")
_CJS_OBJECT_SHORTHAND = re.compile(rf"^({_IDENT})$")

# Python: a statically-literal ``__all__`` assignment.
_PY_ALL_HEAD = re.compile(r"(?m)^__all__\s*(?::[^=\n]+)?\s*(?:=|\+=)\s*(.*)$")
_PY_ALL_STRING = re.compile(r"""(['"])([A-Za-z_]\w*)\1""")


@dataclass(frozen=True)
class LexicalProof:
    """Deterministically ordered declarations proven from visible source."""

    functions: tuple[tuple[str, int], ...] = ()
    classes: tuple[tuple[str, int], ...] = ()
    exports: tuple[tuple[str, int], ...] = ()

    def any(self) -> bool:
        return bool(self.functions or self.classes or self.exports)

    def names(self, bucket: str) -> tuple[tuple[str, int], ...]:
        return {
            "functions": self.functions,
            "classes": self.classes,
            "exports": self.exports,
        }[bucket]


def _visible_lines(source: str) -> Iterable[tuple[int, str]]:
    """Yield ``(line_start_byte, code_only_line)`` for every physical line,
    with block comments, line comments, and multi-line string/template bodies
    removed. Deliberately conservative: a line whose string state cannot be
    resolved is yielded empty so no declaration is proven from it."""
    offset = 0
    in_block_comment = False
    in_backtick = False
    in_triple_single = False
    in_triple_double = False
    for raw_line in source.splitlines(keepends=True):
        line_start = offset
        offset += len(raw_line.encode("utf-8"))
        line = raw_line.rstrip("\r\n")

        if in_triple_single or in_triple_double:
            marker = "'''" if in_triple_single else '"""'
            end = line.find(marker)
            if end == -1:
                continue
            line = line[end + 3 :]
            in_triple_single = in_triple_double = False

        code_chars: list[str] = []
        index = 0
        length = len(line)
        suppressed = in_block_comment or in_backtick
        while index < length:
            two = line[index : index + 2]
            three = line[index : index + 3]
            if in_block_comment:
                if two == "*/":
                    in_block_comment = False
                    index += 2
                    continue
                index += 1
                continue
            if in_backtick:
                if line[index] == "`":
                    in_backtick = False
                index += 1
                continue
            if two == "//":
                break
            if line[index] == "#":
                break
            if two == "/*":
                in_block_comment = True
                index += 2
                continue
            if three in ("'''", '"""'):
                # Opening a triple-quote on this line. If it also closes on the
                # same line, drop the body; otherwise enter multi-line state.
                closing = line.find(three, index + 3)
                if closing == -1:
                    if three == "'''":
                        in_triple_single = True
                    else:
                        in_triple_double = True
                    index = length
                    continue
                index = closing + 3
                continue
            if line[index] == "`":
                in_backtick = True
                index += 1
                continue
            if line[index] in ("'", '"'):
                quote = line[index]
                index += 1
                while index < length and line[index] != quote:
                    if line[index] == "\\":
                        index += 1
                    index += 1
                index += 1
                continue
            code_chars.append(line[index])
            index += 1
        yield line_start, ("" if suppressed else "".join(code_chars))


def _binding_bucket(annotation: str | None, initializer: str) -> str | None:
    """Return the proven bucket for a ``const/let/var NAME [:T] = INIT`` line,
    or ``None`` when the initializer proves nothing."""
    init = initializer.strip()
    if _JS_ARROW_INIT.match(init) or _JS_FUNC_INIT.match(init):
        return "functions"
    if annotation is not None and _REACT_COMPONENT_ANNOTATION.match(
        annotation.strip()
    ):
        # A typed React function component is a function/component, never a
        # class -- but only when an initializer is actually present, which the
        # binding regex already required.
        return "functions"
    return None


def _balanced_slice(
    text: str, open_char: str, close_char: str
) -> tuple[str, str, bool]:
    """Return ``(body, rest, ok)`` for the bracket run that starts at the first
    *open_char* in *text*: *body* is what sits between the outermost brackets,
    *rest* is everything after the closing bracket, and *ok* is False when the
    run never closes or nests another bracket (a nested collection is not a
    flat literal we will trust)."""
    start = text.find(open_char)
    if start == -1:
        return "", "", False
    depth = 0
    quote: str | None = None
    for index in range(start, len(text)):
        char = text[index]
        if quote is not None:
            if char == "\\":
                continue
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            continue
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], text[index + 1 :], True
        elif char in "[](){}":
            # any other bracket inside -> not a flat literal
            return "", "", False
    return "", "", False


def _prove_python_all_exports(source: str) -> list[tuple[str, int]]:
    """Names listed in a statically-literal ``__all__`` list/tuple.

    A single non-string element, concatenation (``+``), comprehension, call, or
    trailing operator voids the whole proof (a dynamically built ``__all__`` is
    not statically provable).
    """
    if "__all__" not in source:
        return []
    proven: list[tuple[str, int]] = []
    for match in _PY_ALL_HEAD.finditer(source):
        first_line_rhs = match.group(1)
        lead = len(first_line_rhs) - len(first_line_rhs.lstrip())
        rhs = first_line_rhs.lstrip()
        if not rhs or rhs[0] not in "[(":
            return []
        open_char = rhs[0]
        close_char = "]" if open_char == "[" else ")"
        tail = source[match.start(1) + lead :]
        body, rest, ok = _balanced_slice(tail, open_char, close_char)
        if not ok:
            return []
        # nothing but whitespace / a comment may follow the literal on its line
        rest_line = rest.split("\n", 1)[0].strip()
        if rest_line and not rest_line.startswith("#") and rest_line not in (";",):
            return []
        residue = _PY_ALL_STRING.sub("", body)
        if residue.strip(" \t\r\n,"):
            # something other than string literals + separators -> not literal
            return []
        offset = len(source[: match.start()].encode("utf-8"))
        for item in _PY_ALL_STRING.finditer(body):
            proven.append((item.group(2), offset))
    return proven


def prove_lexical_declarations(source: str, language: str) -> LexicalProof:
    """Prove function/class/export declarations from *source* for *language*.

    Only the plan's initially-required languages are recognized. Comments,
    strings, template literals, property assignments, calls, imports, dynamic
    or computed exports, and ambiguous constructs prove nothing. Function and
    class identity is occurrence-based: two real same-name declarations at
    different offsets are two proofs, never one.
    """
    lang = (language or "").lower()
    if lang not in _LEXICAL_PROOF_LANGUAGES or not isinstance(source, str):
        return LexicalProof()

    functions: list[tuple[str, int]] = []
    classes: list[tuple[str, int]] = []
    exports: list[tuple[str, int]] = []
    seen_ex: set[str] = set()
    export_pos = 0

    is_python = lang == "python"

    def _add_decl(bucket: list, name: str, offset: int) -> None:
        key = normalize_declaration_name(name, lang)
        if key:
            bucket.append((key, offset))

    def _add_export(name: str) -> None:
        # Every proven export gets its own strictly increasing occurrence
        # position, assigned in the exact order names appear in canonical
        # source (top to bottom, and left to right within one line). The
        # position is an internal ordering key only -- it never leaves this
        # module. The first occurrence of a name wins; a later duplicate keeps
        # that first position.
        nonlocal export_pos
        key = normalize_declaration_name(name, lang)
        if key and key not in seen_ex:
            seen_ex.add(key)
            exports.append((key, export_pos))
            export_pos += 1

    visible = list(_visible_lines(source))

    if is_python:
        # Only trust an ``__all__`` whose line is real code (not inside a
        # string/comment body, which ``_visible_lines`` blanks out).
        all_line_starts = {
            start
            for start, code in visible
            if code.lstrip().startswith("__all__")
        }
        for name, offset in _prove_python_all_exports(source):
            if offset in all_line_starts:
                _add_export(name)

    for line_start, code in visible:
        stripped = code.strip()
        if not stripped:
            continue

        if is_python:
            match = _PY_DEF.match(stripped)
            if match:
                _add_decl(functions, match.group(1), line_start)
                continue
            match = _PY_CLASS.match(stripped)
            if match:
                _add_decl(classes, match.group(1), line_start)
            continue

        # JavaScript / TypeScript family.
        match = _JS_FUNC_DECL.match(stripped)
        if match:
            _add_decl(functions, match.group(1), line_start)
            if stripped.startswith("export"):
                _add_export(match.group(1))
            continue
        match = _JS_CLASS_DECL.match(stripped)
        if match:
            _add_decl(classes, match.group(1), line_start)
            if stripped.startswith("export"):
                _add_export(match.group(1))
            continue
        match = _JS_BINDING.match(stripped)
        if match:
            bucket = _binding_bucket(match.group("annotation"), match.group("init"))
            if bucket == "functions":
                _add_decl(functions, match.group(1), line_start)
            # An explicit ``export const/let/var NAME = ...`` is a named module
            # export regardless of what the initializer is (function, component,
            # or plain data).
            if stripped.startswith("export"):
                _add_export(match.group(1))
            continue
        block = _JS_NAMED_EXPORT_BLOCK.match(stripped)
        if block:
            for raw in block.group(1).split(","):
                token = raw.strip()
                if not token or token.startswith("*"):
                    continue
                exported = token.split(" as ")[-1].strip()
                if re.fullmatch(_IDENT, exported):
                    _add_export(exported)
            continue
        star_as = _JS_EXPORT_STAR_AS.match(stripped)
        if star_as:
            _add_export(star_as.group(1))
            continue
        default_named = _JS_EXPORT_DEFAULT_NAME.match(stripped)
        if default_named:
            _add_export(default_named.group(1))
            continue
        whole_object = _CJS_WHOLE_OBJECT.match(stripped)
        if whole_object:
            inner = whole_object.group(1)
            if "..." not in inner and "[" not in inner:
                for part in inner.split(","):
                    piece = part.strip()
                    if not piece:
                        continue
                    keyed = _CJS_OBJECT_MEMBER.match(piece)
                    shorthand = _CJS_OBJECT_SHORTHAND.match(piece)
                    if keyed:
                        _add_export(keyed.group(1))
                    elif shorthand:
                        _add_export(shorthand.group(1))
            continue
        whole_name = _CJS_WHOLE_NAME.match(stripped)
        if whole_name:
            _add_export(whole_name.group(1))
            continue
        prop_export = _CJS_PROP_EXPORT.match(stripped)
        if prop_export:
            _add_export(prop_export.group(1))
            continue

    return LexicalProof(
        functions=tuple(functions),
        classes=tuple(classes),
        exports=tuple(exports),
    )


# ---------------------------------------------------------------------------
# Structure truth + reconciliation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructureTruth:
    """The immutable source-derived structural evidence for one file."""

    rel_path: str
    language: str
    source: str
    symbols: tuple[SymbolFact, ...] = ()
    structural_mode: str = "lexical"

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbols", tuple(self.symbols))

    @property
    def has_symbol_authority(self) -> bool:
        return bool(self.symbols)

    @classmethod
    def from_source(
        cls, rel_path: str, language: str, source: str
    ) -> "StructureTruth":
        """Build truth by extracting structure from the immutable planned
        *source* string (never a fresh filesystem read)."""
        from codedoc.parser.tree_sitter_structure import extract_structure

        try:
            result = extract_structure(rel_path, language, source)
            symbols = result.symbols
            mode = result.structural_mode
        except Exception:
            symbols = ()
            mode = "lexical"
        return cls(
            rel_path=normalize_rel_path(rel_path),
            language=language,
            source=source if isinstance(source, str) else "",
            symbols=symbols,
            structural_mode=mode,
        )

    @classmethod
    def from_split_plan(
        cls, rel_path: str, language: str, division_plan: object
    ) -> "StructureTruth":
        """Reuse the parser-owned facts split planning already computed."""
        chunks = getattr(division_plan, "chunks", ()) or ()
        source = "".join(
            chunk.payload for chunk in chunks if isinstance(getattr(chunk, "payload", None), str)
        )
        return cls(
            rel_path=normalize_rel_path(rel_path),
            language=language,
            source=source,
            symbols=tuple(getattr(division_plan, "symbols", ()) or ()),
            structural_mode=getattr(division_plan, "structural_mode", "lexical"),
        )


@dataclass
class _AuthDecl:
    bucket: str
    order: int
    identity: str
    full_name: str
    short_name: str
    signature: str = ""
    referenced: bool = False
    described: bool = False
    description: str = ""


@dataclass(frozen=True)
class ReconciledArrays:
    functions: tuple[dict, ...] = ()
    classes: tuple[dict, ...] = ()
    exports: tuple[str, ...] = ()
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    enforced: bool = False


def _model_name(item: object) -> str | None:
    if isinstance(item, Mapping):
        raw = item.get("name")
        return raw if isinstance(raw, str) and raw.strip() else None
    return None


def _model_description(item: object) -> str:
    if isinstance(item, Mapping):
        raw = item.get("description")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return ""


def _model_signature(item: object) -> str:
    if isinstance(item, Mapping):
        raw = item.get("signature")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return ""


_SIGNATURE_WS = re.compile(r"\s+")


def _signatures_compatible(model_sig: str, auth_sig: str) -> bool:
    """Whitespace-insensitive containment either way -- the same loose test the
    split fact ledger uses to line a model signature up with a parser one."""
    left = _SIGNATURE_WS.sub("", model_sig)
    right = _SIGNATURE_WS.sub("", auth_sig)
    return bool(left) and bool(right) and (left in right or right in left)


def _bounded_count(value: int) -> int:
    return max(0, min(int(value), _MAX_DIAGNOSTIC_COUNT))


def reconcile_structural_arrays(
    functions: Sequence[object],
    classes: Sequence[object],
    exports: Sequence[object],
    *,
    truth: StructureTruth,
) -> ReconciledArrays:
    """Return the source-backed ``functions`` / ``classes`` / ``exports`` for
    one file, plus a value-safe diagnostic summary.

    A model item is published only when it names a declaration the source
    proves; its only contribution is a bounded description to exactly one
    compatible authoritative declaration. Kind comes from the source, never the
    model. Two model reports of one declaration collapse; two distinct
    declarations with the same name stay distinct, and an ambiguous same-name
    report with no disambiguating signature attaches no description rather than
    guessing. Output order is source order -- never response or split-leaf
    order. There is no language for which unproven model items pass through.
    """
    lang = truth.language
    proof = prove_lexical_declarations(truth.source, lang)

    diagnostics: dict[str, object] = {
        "path": truth.rel_path,
        "structural_mode": truth.structural_mode,
    }

    # Authoritative declaration index. Parser symbols first -- they own kind,
    # identity, signature, and source order; a symbol whose kind maps to no
    # publishable bucket is not an authoritative function or class and is
    # skipped. The conservative lexical recognizer then supplements any
    # (name, bucket) no symbol already provided, keeping occurrence identity so
    # two real same-name declarations stay two declarations.
    auth: list[_AuthDecl] = []
    covered: set[tuple[str, str]] = set()
    for symbol in truth.symbols:
        bucket = parser_kind_bucket(symbol.kind)
        if bucket is None:
            continue
        short = _short_name(symbol.qualified_name, lang)
        full = normalize_declaration_name(symbol.qualified_name, lang)
        auth.append(
            _AuthDecl(
                bucket=bucket,
                order=symbol.range.start_byte,
                identity=symbol.symbol_id,
                full_name=full,
                short_name=short,
                signature=getattr(symbol, "signature", "") or "",
            )
        )
        covered.add((short, bucket))
        covered.add((full, bucket))
    for bucket in ("functions", "classes"):
        for name, offset in proof.names(bucket):
            if (name, bucket) in covered:
                continue
            auth.append(
                _AuthDecl(
                    bucket=bucket,
                    order=offset,
                    identity=f"lexical:{bucket}:{name}:{offset}",
                    full_name=name,
                    short_name=name,
                )
            )

    by_short: dict[str, list[_AuthDecl]] = {}
    by_identity: dict[str, _AuthDecl] = {}
    for decl in auth:
        labels = [decl.short_name]
        if decl.full_name != decl.short_name:
            labels.append(decl.full_name)
        for label in labels:
            by_short.setdefault(label, []).append(decl)
        by_identity[decl.identity] = decl

    rejected = {"functions": 0, "classes": 0, "exports": 0}
    matched = {"functions": 0, "classes": 0, "exports": 0}

    def _eligible(
        candidates: list[_AuthDecl], model_bucket: str
    ) -> list[_AuthDecl]:
        """The source-backed declarations a model claim in *model_bucket* may
        publish, applying source-kind authority:

        * if any same-name candidate is already in the claimed bucket, only
          that compatible bucket is eligible -- candidates in the other bucket
          are never published off this claim (the ``def Same`` / ``class Same``
          case);
        * else, when exactly one same-name candidate exists in a single other
          bucket, the proven source kind overrides the model's wrong
          classification (F-5: a class claim for a proven ``React.FC`` binding
          publishes the function/component);
        * else the kind correction itself is ambiguous -- publish nothing.
        """
        same_bucket = [d for d in candidates if d.bucket == model_bucket]
        if same_bucket:
            return same_bucket
        other_buckets = {d.bucket for d in candidates}
        if len(candidates) == 1 and len(other_buckets) == 1:
            return list(candidates)
        return []

    def _pick(eligible: list[_AuthDecl], model_sig: str) -> _AuthDecl | None:
        """The one eligible declaration a description may attach to, or ``None``
        when the eligible set is an ambiguous overload group nothing isolates."""
        if len(eligible) == 1:
            # exactly one eligible declaration: a duplicate report about it
            # simply consolidates here.
            return eligible[0]
        if model_sig:
            sig_hits = [
                d
                for d in eligible
                if d.signature and _signatures_compatible(model_sig, d.signature)
            ]
            if len(sig_hits) == 1:
                return sig_hits[0]
            free = [d for d in sig_hits if not d.described]
            if len(free) == 1:
                return free[0]
        # >1 eligible overloads and no signature that isolates one: do not
        # guess which overload the description belongs to.
        return None

    def _consume(items: Sequence[object], model_bucket: str) -> None:
        for item in items:
            name = _model_name(item)
            if name is None:
                rejected[model_bucket] += 1
                continue
            key = normalize_declaration_name(name, lang)
            short = _short_name(name, lang)
            candidates = by_short.get(key) or by_short.get(short) or []
            # Determine the kind-compatible eligible set *before* marking
            # anything referenced, so an incompatible same-name declaration in
            # the other bucket is never dragged into the output.
            eligible = _eligible(candidates, model_bucket)
            if not eligible:
                rejected[model_bucket] += 1
                continue
            matched[model_bucket] += 1
            for decl in eligible:
                decl.referenced = True
            description = _model_description(item)
            if not description:
                continue
            target = _pick(eligible, _model_signature(item))
            if target is not None and not target.description:
                target.description = description
                target.described = True

    _consume(functions, "functions")
    _consume(classes, "classes")

    def _publish(bucket: str) -> list[dict]:
        return [
            {
                "name": d.full_name,
                **({"description": d.description} if d.description else {}),
            }
            for d in sorted(
                (
                    d
                    for d in by_identity.values()
                    if d.bucket == bucket and d.referenced
                ),
                key=lambda d: (d.order, d.identity),
            )
        ]

    published_functions = _publish("functions")
    published_classes = _publish("classes")

    # Exports are never SymbolFacts. Only the conservative recognizer's closed
    # proof set admits an export (ESM export declarations, named + namespace
    # re-exports, CommonJS ``module.exports`` / ``exports.NAME``, a statically
    # literal Python ``__all__``). A language with no such proof publishes no
    # exports; there is no passthrough.
    # Each proven export carries its canonical-source occurrence position
    # (assigned by the recognizer). The published list is ordered by that
    # position alone -- never by export name -- so response order and
    # alphabetics cannot reorder it. First occurrence of a name wins; a later
    # duplicate model report of the same proven export collapses onto it.
    proven_exports: dict[str, int] = {}
    for name, position in proof.exports:
        proven_exports.setdefault(name, position)
    published_exports: list[tuple[str, int]] = []
    seen_exports: set[str] = set()
    for raw in exports:
        if not isinstance(raw, str) or not raw.strip():
            rejected["exports"] += 1
            continue
        key = normalize_declaration_name(raw, lang)
        if key in seen_exports:
            continue
        if key in proven_exports:
            seen_exports.add(key)
            matched["exports"] += 1
            published_exports.append((raw.strip(), proven_exports[key]))
        else:
            rejected["exports"] += 1
    published_exports.sort(key=lambda pair: pair[1])

    diagnostics.update(
        {
            "authoritative_declarations": len(auth),
            "matched": {k: _bounded_count(v) for k, v in matched.items()},
            "rejected_unmatched": {
                k: _bounded_count(v) for k, v in rejected.items()
            },
        }
    )
    return ReconciledArrays(
        functions=tuple(published_functions),
        classes=tuple(published_classes),
        exports=tuple(name for name, _offset in published_exports),
        diagnostics=diagnostics,
        enforced=True,
    )


def strip_transient_symbol_fields(items: object) -> list[dict]:
    """Project each symbol mapping to the public ``{name, description?}`` shape.

    Removes any transient internal field the reconciliation-mode response
    cleaner carried this far -- a per-symbol ``signature`` used only to line a
    model description up with the right same-name overload, and defensively any
    id / range / provenance key. Order and multiplicity are preserved; nothing
    else about the list changes. Used on the routes where reconciliation itself
    did not rebuild the arrays (a failed sub-agent, a missing truth) so the
    signature never reaches a record, a checkpoint, or another agent's prompt.
    """
    projected: list[dict] = []
    for item in items or ():
        if not isinstance(item, Mapping):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        shaped: dict = {"name": name}
        description = item.get("description")
        if isinstance(description, str) and description.strip():
            shaped["description"] = description
        projected.append(shaped)
    return projected


def apply_structural_reconciliation(
    cleaned: Mapping[str, object], truth: StructureTruth
) -> dict:
    """Return a shallow copy of *cleaned* whose ``functions`` / ``classes`` /
    ``exports`` are the reconciled, source-backed arrays. Every other field is
    untouched. No diagnostic or provenance is added to the public mapping.
    """
    result = dict(cleaned)
    reconciled = reconcile_structural_arrays(
        result.get("functions", []) or [],
        result.get("classes", []) or [],
        result.get("exports", []) or [],
        truth=truth,
    )
    result["functions"] = [dict(item) for item in reconciled.functions]
    result["classes"] = [dict(item) for item in reconciled.classes]
    result["exports"] = list(reconciled.exports)
    rejected = reconciled.diagnostics.get("rejected_unmatched")
    if isinstance(rejected, Mapping) and any(rejected.values()):
        # Value-safe: normalized path + counts only; never a name, description,
        # signature, id, range, prompt, or response value (section 5.4).
        logger.debug(
            "structural reconciliation dropped unproven model items for %s: %s",
            truth.rel_path,
            {key: int(value) for key, value in rejected.items()},
        )
    return result
