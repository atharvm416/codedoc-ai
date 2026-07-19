"""Tests for the React/TSX parser."""

from tests.support.fixture_paths import PROJECT_FIXTURES, SOURCE_FIXTURES

class TestReactParser:
    def test_extracts_relative_imports(self, tmp_path):
        f = tmp_path / "App.tsx"
        f.write_text("import React from 'react';\nimport Header from './Header';\nimport { router } from '../router';\n")
        from codedoc.parser.react_parser import parse
        result = parse(f)
        assert "./Header" in result
        assert "../router" in result
        assert "react" not in result  # npm package excluded

    def test_fixture_react_app(self):
        from codedoc.parser.react_parser import parse
        result = parse(PROJECT_FIXTURES / "react_app" / "App.tsx")
        assert isinstance(result, list)

    def test_fixture_react_sample(self):
        from codedoc.parser.react_parser import parse
        result = parse(SOURCE_FIXTURES / "react_sample.tsx")
        assert isinstance(result, list)
