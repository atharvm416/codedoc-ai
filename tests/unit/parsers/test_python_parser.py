"""Tests for the Python parser."""

import pytest

from tests.support.fixture_paths import PROJECT_FIXTURES

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
        result = parse(PROJECT_FIXTURES / "python_app" / "main.py")
        assert isinstance(result, list)

    def test_syntax_error_raises_parse_error(self, tmp_path):
        f = tmp_path / "bad.py"
        f.write_text("def broken(\n")
        from codedoc.parser.python_parser import parse
        from codedoc.utils.errors import ParseError
        with pytest.raises(ParseError):
            parse(f)
