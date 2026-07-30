"""Declarative grammar and declaration rules for optional structure extraction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LanguageSpec:
    language: str
    grammar_name: str
    declaration_node_types: tuple[str, ...]
    name_fields: tuple[str, ...] = ("name",)


LANGUAGE_SPECS: dict[str, LanguageSpec] = {
    "python": LanguageSpec(
        "python", "python", ("function_definition", "class_definition")
    ),
    "javascript": LanguageSpec(
        "javascript",
        "javascript",
        (
            "function_declaration",
            "class_declaration",
            "method_definition",
            "generator_function_declaration",
        ),
    ),
    "jsx": LanguageSpec(
        "jsx",
        "javascript",
        ("function_declaration", "class_declaration", "method_definition"),
    ),
    "typescript": LanguageSpec(
        "typescript",
        "typescript",
        (
            "function_declaration",
            "class_declaration",
            "method_definition",
            "interface_declaration",
            "type_alias_declaration",
        ),
    ),
    "tsx": LanguageSpec(
        "tsx",
        "tsx",
        (
            "function_declaration",
            "class_declaration",
            "method_definition",
            "interface_declaration",
            "type_alias_declaration",
        ),
    ),
    "dart": LanguageSpec(
        "dart",
        "dart",
        (
            "function_signature",
            "method_signature",
            "class_definition",
            "enum_declaration",
        ),
    ),
    "java": LanguageSpec(
        "java",
        "java",
        (
            "method_declaration",
            "constructor_declaration",
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
        ),
    ),
    "csharp": LanguageSpec(
        "csharp",
        "csharp",
        (
            "method_declaration",
            "constructor_declaration",
            "class_declaration",
            "struct_declaration",
            "interface_declaration",
            "enum_declaration",
        ),
    ),
    "html": LanguageSpec("html", "html", ("element", "script_element")),
    "kotlin": LanguageSpec(
        "kotlin",
        "kotlin",
        ("function_declaration", "class_declaration", "object_declaration"),
    ),
    "swift": LanguageSpec(
        "swift",
        "swift",
        (
            "function_declaration",
            "class_declaration",
            "protocol_declaration",
            "struct_declaration",
            "enum_declaration",
        ),
    ),
    "go": LanguageSpec(
        "go",
        "go",
        ("function_declaration", "method_declaration", "type_declaration"),
    ),
    "ruby": LanguageSpec(
        "ruby", "ruby", ("method", "singleton_method", "class", "module")
    ),
    "rust": LanguageSpec(
        "rust",
        "rust",
        (
            "function_item",
            "struct_item",
            "enum_item",
            "trait_item",
            "impl_item",
        ),
    ),
    "c": LanguageSpec(
        "c", "c", ("function_definition", "struct_specifier", "enum_specifier")
    ),
    "cpp": LanguageSpec(
        "cpp",
        "cpp",
        (
            "function_definition",
            "class_specifier",
            "struct_specifier",
            "namespace_definition",
            "enum_specifier",
        ),
    ),
}


def spec_for(language: str) -> LanguageSpec | None:
    if not isinstance(language, str):
        return None
    return LANGUAGE_SPECS.get(language.lower())
