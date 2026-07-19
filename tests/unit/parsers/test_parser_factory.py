"""Tests for parser dispatch."""

from tests.support.fixture_paths import PROJECT_FIXTURES

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
        p = PROJECT_FIXTURES / "java_app" / "Main.java"
        descriptor = {"path": p, "rel_path": "Main.java", "language": "java", "extension": ".java"}
        result = parse_file(descriptor)
        assert isinstance(result, list)
