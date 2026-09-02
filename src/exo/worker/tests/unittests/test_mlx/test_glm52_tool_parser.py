"""Unit tests for the GLM-5.2 tool parser.

The GLM-5.2 chat template uses angle-bracket special-token delimiters
that decode (via ``tokenizer.decode``) to plain ASCII strings WITHOUT
the pipe character, e.g. ``<tool_call>`` / ``〈/tool_call〉`` /
``〈arg_key〉`` / ``〈/arg_key〉`` / ``〈arg_value〉`` / ``〈/arg_value〉``.
These tests verify the glm52 parser handles that format correctly.
"""

from exo.worker.engines.mlx.tool_parsers.glm52 import (
    parse_tool_call as glm52_parse,
)
from exo.worker.engines.mlx.tool_parsers.glm52 import (
    tool_call_end as glm52_end,
)
from exo.worker.engines.mlx.tool_parsers.glm52 import (
    tool_call_start as glm52_start,
)

# Plain ASCII angle-bracket delimiters (no pipe) matching tokenizer.decode.
_AK_OPEN = chr(60) + "arg_key" + chr(62)
_AK_CLOSE = chr(60) + "/arg_key" + chr(62)
_AV_OPEN = chr(60) + "arg_value" + chr(62)
_AV_CLOSE = chr(60) + "/arg_value" + chr(62)

EXPECTED_START = chr(60) + "tool_call" + chr(62)
EXPECTED_END = chr(60) + "/tool_call" + chr(62)


class TestGlm52ToolParserDelimiters:
    """Tests for the tool call delimiter constants."""

    def test_tool_call_start(self) -> None:
        assert glm52_start == EXPECTED_START

    def test_tool_call_end(self) -> None:
        assert glm52_end == EXPECTED_END


class TestGlm52ToolParsing:
    """Tests for parsing GLM-5.2 format tool calls.

    The input text has already had TOOL_CALL_START and TOOL_CALL_END
    stripped by make_mlx_parser, so it starts with the function name.
    """

    def test_single_argument(self) -> None:
        text = (
            "get_weather"
            + _AK_OPEN
            + "location"
            + _AK_CLOSE
            + _AV_OPEN
            + "San Francisco"
            + _AV_CLOSE
        )
        result = glm52_parse(text)
        assert result["name"] == "get_weather"
        assert result["arguments"]["location"] == "San Francisco"

    def test_multiple_arguments(self) -> None:
        text = (
            "get_weather"
            + _AK_OPEN
            + "location"
            + _AK_CLOSE
            + _AV_OPEN
            + "San Francisco"
            + _AV_CLOSE
            + _AK_OPEN
            + "unit"
            + _AK_CLOSE
            + _AV_OPEN
            + "celsius"
            + _AV_CLOSE
        )
        result = glm52_parse(text)
        assert result["name"] == "get_weather"
        assert result["arguments"]["location"] == "San Francisco"
        assert result["arguments"]["unit"] == "celsius"

    def test_empty_arguments(self) -> None:
        text = "get_weather"
        result = glm52_parse(text)
        assert result["name"] == "get_weather"
        assert result["arguments"] == {}


class TestGlm52ToolParsingEdgeCases:
    """Edge case tests for the GLM-5.2 tool parser."""

    def test_no_delimiters(self) -> None:
        text = "get_weather"
        result = glm52_parse(text)
        assert result["name"] == "get_weather"
        assert result["arguments"] == {}

    def test_string_arg_deserialization(self) -> None:
        text = "fn" + _AK_OPEN + "key" + _AK_CLOSE + _AV_OPEN + "value" + _AV_CLOSE
        result = glm52_parse(text)
        assert result["name"] == "fn"
        assert result["arguments"]["key"] == "value"
