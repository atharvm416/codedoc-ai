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

class TestGoParserA3:
    """A3: Go parser must not treat arbitrary string literals as imports."""

    def test_single_and_block_imports_only(self, tmp_path):
        f = tmp_path / "main.go"
        f.write_text(
            'package main\n'
            'import "fmt"\n'
            'import (\n'
            '    "os"\n'
            '    alias "github.com/x/y"\n'
            '    _ "github.com/z/driver"\n'
            ')\n'
            'func main() {\n'
            '    fmt.Println("hello world")\n'  # must NOT become an import
            '    s := "not/an/import"\n'        # must NOT become an import
            '}\n',
            encoding="utf-8",
        )
        from codedoc.parser.generic_parser import parse
        result = parse(f, "go")
        assert "fmt" in result
        assert "os" in result
        assert "github.com/x/y" in result
        assert "github.com/z/driver" in result
        assert "hello world" not in result
        assert "not/an/import" not in result


class TestHtmlParserA3:
    """A3: HTML parser must not treat CSS <link href> as a code import."""

    def test_script_src_kept_link_href_dropped(self, tmp_path):
        f = tmp_path / "index.html"
        f.write_text(
            '<html><head>\n'
            '<link rel="stylesheet" href="styles/app.css">\n'
            '<link rel="icon" href="favicon.ico">\n'
            '<script src="js/app.js"></script>\n'
            '</head></html>\n',
            encoding="utf-8",
        )
        from codedoc.parser.generic_parser import parse
        result = parse(f, "html")
        assert "js/app.js" in result
        assert "styles/app.css" not in result
        assert "favicon.ico" not in result


class TestGoParserA3Comments:
    """A3 follow-up: Go comments must not yield false imports; raw-string
    (backtick) import paths are accepted."""

    def test_comments_inside_import_block_ignored(self, tmp_path):
        f = tmp_path / "main.go"
        f.write_text(
            'package main\n'
            'import (\n'
            '    "fmt" // formatting\n'
            '    // "os" is intentionally commented out\n'
            '    /* "net/http" disabled */\n'
            '    "strings"\n'
            ')\n',
            encoding="utf-8",
        )
        from codedoc.parser.generic_parser import parse
        result = parse(f, "go")
        assert "fmt" in result
        assert "strings" in result
        assert "os" not in result
        assert "net/http" not in result

    def test_commented_out_import_block_ignored(self, tmp_path):
        f = tmp_path / "main.go"
        f.write_text(
            'package main\n'
            '// import (\n'
            '//     "fmt"\n'
            '// )\n'
            'import "strings"\n',
            encoding="utf-8",
        )
        from codedoc.parser.generic_parser import parse
        result = parse(f, "go")
        assert "strings" in result
        assert "fmt" not in result

    def test_backtick_import_path_supported(self, tmp_path):
        f = tmp_path / "main.go"
        f.write_text(
            'package main\n'
            'import `fmt`\n'
            'import (\n'
            '    `strings`\n'
            ')\n',
            encoding="utf-8",
        )
        from codedoc.parser.generic_parser import parse
        result = parse(f, "go")
        assert "fmt" in result
        assert "strings" in result


class TestGoParserA3Escapes:
    """Finding 3: interpreted Go string escapes are decoded; raw (backtick)
    strings are left literal per the Go spec. The .go content is built from
    chr() so Python's own parser cannot pre-decode the escape under test."""

    def test_hex_escape_decoded(self, tmp_path):
        nl, bs, q = chr(10), chr(92), chr(34)
        content = "package main" + nl + "import " + q + "f" + bs + "x6dt" + q + nl
        f = tmp_path / "main.go"
        f.write_text(content, encoding="utf-8")
        from codedoc.parser.generic_parser import parse
        result = parse(f, "go")
        assert "fmt" in result
        assert ("f" + bs + "x6dt") not in result

    def test_unicode_escape_decoded(self, tmp_path):
        nl, bs, q = chr(10), chr(92), chr(34)
        content = "package main" + nl + "import " + q + bs + "u0066mt" + q + nl
        f = tmp_path / "main.go"
        f.write_text(content, encoding="utf-8")
        from codedoc.parser.generic_parser import parse
        assert "fmt" in parse(f, "go")

    def test_utf8_hex_bytes_equal_unicode_escape_and_literal(self, tmp_path):
        nl, bs, q = chr(10), chr(92), chr(34)
        escaped_bytes = "caf" + bs + "xc3" + bs + "xa9"
        unicode_escape = "caf" + bs + "u00e9"
        literal = "caf" + chr(0xE9)
        content = (
            "package main" + nl
            + "import (" + nl
            + q + escaped_bytes + q + nl
            + q + unicode_escape + q + nl
            + q + literal + q + nl
            + ")" + nl
        )
        f = tmp_path / "main.go"
        f.write_text(content, encoding="utf-8")
        from codedoc.parser.generic_parser import parse
        assert parse(f, "go") == [literal]

    def test_utf8_hex_bytes_equal_unicode_escape_for_multibyte_rune(self, tmp_path):
        nl, bs, q = chr(10), chr(92), chr(34)
        escaped_bytes = bs + "xe6" + bs + "x97" + bs + "xa5"
        unicode_escape = bs + "u65e5"
        literal = chr(0x65E5)
        content = (
            "package main" + nl
            + "import (" + nl
            + q + escaped_bytes + q + nl
            + q + unicode_escape + q + nl
            + q + literal + q + nl
            + ")" + nl
        )
        f = tmp_path / "main.go"
        f.write_text(content, encoding="utf-8")
        from codedoc.parser.generic_parser import parse
        assert parse(f, "go") == [literal]

    def test_utf8_octal_bytes_equal_unicode_literal(self, tmp_path):
        nl, bs, q = chr(10), chr(92), chr(34)
        escaped_bytes = "caf" + bs + "303" + bs + "251"
        literal = "caf" + chr(0xE9)
        content = (
            "package main" + nl
            + "import (" + nl
            + q + escaped_bytes + q + nl
            + q + literal + q + nl
            + ")" + nl
        )
        f = tmp_path / "main.go"
        f.write_text(content, encoding="utf-8")
        from codedoc.parser.generic_parser import parse
        assert parse(f, "go") == [literal]

    def test_escaped_unicode_import_resolves_local_file(self, tmp_path):
        nl, bs, q = chr(10), chr(92), chr(34)
        name = "caf" + chr(0xE9)
        source = "package main" + nl + "import " + q + "pkg/caf" + bs + "xc3" + bs + "xa9" + q + nl
        main_go = tmp_path / "main.go"
        main_go.write_text(source, encoding="utf-8")
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / f"{name}.go").write_text("package cafe\n", encoding="utf-8")

        from codedoc.core.graph import resolve_import
        from codedoc.parser.generic_parser import parse

        imports = parse(main_go, "go")
        all_files = {"main.go", f"pkg/{name}.go"}
        assert imports == [f"pkg/{name}"]
        assert resolve_import(imports[0], "main.go", all_files, tmp_path) == f"pkg/{name}.go"

    def test_normal_path_with_no_escape_unchanged(self, tmp_path):
        nl, q = chr(10), chr(34)
        content = "package main" + nl + "import " + q + "github.com/x/y" + q + nl
        f = tmp_path / "main.go"
        f.write_text(content, encoding="utf-8")
        from codedoc.parser.generic_parser import parse
        assert "github.com/x/y" in parse(f, "go")

    def test_raw_backtick_string_is_literal(self, tmp_path):
        nl, bs, bt = chr(10), chr(92), chr(96)
        body = bt + "f" + bs + "x6dt" + bt
        content = "package main" + nl + "import " + body + nl
        f = tmp_path / "main.go"
        f.write_text(content, encoding="utf-8")
        from codedoc.parser.generic_parser import parse
        result = parse(f, "go")
        assert ("f" + bs + "x6dt") in result
        assert "fmt" not in result
