"""Tests for all parsers — fully offline, no LLM."""

from pathlib import Path
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


class TestPythonParser:
    def test_extracts_standard_imports(self, tmp_path):
        f = tmp_path / "main.py"
        f.write_text("import os\nimport sys\nfrom pathlib import Path\n")
        from codedoc.parser.python_parser import parse
        result = parse(f)
        assert "os" in result
        assert "sys" in result
        assert "pathlib" in result

    def test_extracts_relative_imports(self, tmp_path):
        f = tmp_path / "app.py"
        f.write_text("from .utils import helper\nfrom ..models import User\n")
        from codedoc.parser.python_parser import parse
        result = parse(f)
        assert ".utils" in result
        assert "..models" in result

    def test_fixture_python_main(self):
        from codedoc.parser.python_parser import parse
        result = parse(FIXTURES / "python_app" / "main.py")
        assert isinstance(result, list)

    def test_syntax_error_raises_parse_error(self, tmp_path):
        f = tmp_path / "bad.py"
        f.write_text("def broken(\n")
        from codedoc.parser.python_parser import parse
        from codedoc.utils.errors import ParseError
        with pytest.raises(ParseError):
            parse(f)


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
        result = parse(FIXTURES / "react_app" / "App.tsx")
        assert isinstance(result, list)

    def test_fixture_react_sample(self):
        from codedoc.parser.react_parser import parse
        result = parse(FIXTURES / "react_sample.tsx")
        assert isinstance(result, list)


class TestGenericParser:
    def test_java(self):
        from codedoc.parser.generic_parser import parse
        result = parse(FIXTURES / "java_app" / "Main.java", "java")
        assert isinstance(result, list)

    def test_dart(self):
        from codedoc.parser.generic_parser import parse
        result = parse(FIXTURES / "flutter_app" / "main.dart", "dart")
        assert isinstance(result, list)


class TestParserFactory:
    def test_routes_python(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("import os\n")
        from codedoc.parser.factory import parse_file
        descriptor = {"path": f, "rel_path": "x.py", "language": "python", "extension": ".py"}
        result = parse_file(descriptor)
        assert "os" in result

    def test_routes_tsx(self, tmp_path):
        f = tmp_path / "x.tsx"
        f.write_text("import Btn from './Btn';\n")
        from codedoc.parser.factory import parse_file
        descriptor = {"path": f, "rel_path": "x.tsx", "language": "tsx", "extension": ".tsx"}
        result = parse_file(descriptor)
        assert "./Btn" in result

    def test_routes_java(self):
        from codedoc.parser.factory import parse_file
        p = FIXTURES / "java_app" / "Main.java"
        descriptor = {"path": p, "rel_path": "Main.java", "language": "java", "extension": ".java"}
        result = parse_file(descriptor)
        assert isinstance(result, list)