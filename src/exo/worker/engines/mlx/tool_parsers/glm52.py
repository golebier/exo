# type: ignore
# ruff: noqa
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 tool parser.

GLM-5.2 emits tool calls using special-token delimiters.  The decoded
token text for these delimiters uses plain ASCII angle brackets and no
pipe character (even though the tokenizer's added_tokens metadata
records them with pipes), so the constants below match the output of
tokenizer.decode.
"""

import json
import re
from typing import Any

# The model's special tokens decode (via tokenizer.decode) to plain ASCII
# angle-bracket strings WITHOUT the pipe character, e.g. <tool_call> / 〈/tool_call〉 /
# 〈arg_key〉 / 〈/arg_key〉 / 〈arg_value〉 / 〈/arg_value〉.  Build them from
# codepoints so the source stays unambiguous.
_TC_OPEN = chr(60) + "tool_call" + chr(62)
_TC_CLOSE = chr(60) + "/tool_call" + chr(62)
_AK_OPEN = chr(60) + "arg_key" + chr(62)
_AK_CLOSE = chr(60) + "/arg_key" + chr(62)
_AV_OPEN = chr(60) + "arg_value" + chr(62)
_AV_CLOSE = chr(60) + "/arg_value" + chr(62)

tool_call_start = _TC_OPEN
tool_call_end = _TC_CLOSE

_ARG_KV_PATTERN = re.compile(
    re.escape(_AK_OPEN) + r"(.*?)" + re.escape(_AK_CLOSE)
    + re.escape(_AV_OPEN) + r"(.*?)" + re.escape(_AV_CLOSE),
    re.DOTALL,
)


def _get_string_arg_names(tool_name: str, tools: list[Any] | None) -> set[str]:
    if tools is None:
        return set()
    for tool in tools:
        func = tool.get("function") if isinstance(tool, dict) else tool
        if not func or func.get("name") != tool_name:
            continue
        params = func.get("parameters") or {}
        properties = params.get("properties") or {}
        return {
            name
            for name, schema in properties.items()
            if isinstance(schema, dict) and schema.get("type") == "string"
        }
    return set()


def _deserialize(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        pass
    try:
        import ast

        return ast.literal_eval(value)
    except Exception:
        pass
    return value


def _normalize_arguments(
    func_name: str,
    arguments: dict[str, Any],
    tools: list[Any] | None,
    string_args: set[str] | None = None,
) -> dict[str, Any]:
    if string_args is None:
        string_args = _get_string_arg_names(func_name, tools)
    normalized: dict[str, Any] = {}
    for key, value in arguments.items():
        if key in string_args:
            normalized[key] = value if isinstance(value, str) else str(value)
            continue
        if isinstance(value, str):
            normalized[key] = _deserialize(value)
        else:
            normalized[key] = value
    return normalized


def parse_tool_call(text: str, tools: list[Any] | None = None) -> dict[str, Any]:
    """Parse a GLM-5.2 tool call string into a name and arguments dict.

    The input text has already had TOOL_CALL_START and TOOL_CALL_END
    stripped by make_mlx_parser, so it starts with the function name,
    followed by arg_key/arg_value pairs.
    """
    first_key = text.find(_AK_OPEN)
    if first_key == -1:
        return dict(name=text.strip(), arguments={})

    func_name = text[:first_key].strip()
    string_args = _get_string_arg_names(func_name, tools)

    arg_dct: dict[str, Any] = {}
    for match in _ARG_KV_PATTERN.finditer(text):
        arg_key = match.group(1).strip()
        arg_val = match.group(2).strip()
        if arg_key not in string_args:
            arg_val = _deserialize(arg_val)
        arg_dct[arg_key] = arg_val

    if not arg_dct:
        return dict(name=func_name, arguments={})

    return dict(
        name=func_name,
        arguments=_normalize_arguments(
            func_name, arg_dct, tools, string_args=string_args
        ),
    )


__all__ = ["parse_tool_call", "tool_call_start", "tool_call_end"]
