"""Minimal direct-Miles SQL agent. Tool execution is local and read-only."""

from __future__ import annotations

import asyncio
import copy
import json
import re
from collections.abc import Iterable

from miles.utils.http_utils import post
from miles.utils.lora import LORA_ADAPTER_NAME
from sglang.srt.entrypoints.openai.protocol import DEFAULT_MODEL_NAME

from .sql_tools import DatabaseUnavailableError, describe_database, ex_score, execute_query, final_sql


SYSTEM = """You are a careful Text-to-SQL agent on an immutable SQLite database.
Use tool calls to inspect the schema and test queries. Finish with one triple-backtick sql block.
Never modify data. Tools use THIS syntax (JSON argument inside parameter):
<tool_call><function=describe_schema><parameter>{"table_name":"table"}</parameter></tool_call>
<tool_call><function=run_sql><parameter>{"sql":"SELECT ..."}</parameter></tool_call>
describe_schema({}) lists all tables, columns, and sample rows (table_name restricts one); run_sql accepts SELECT, WITH, or VALUES, while describe_schema handles schema information. Do not invent other tools."""


def flatten(value):
    if isinstance(value, list):
        for item in value:
            yield from flatten(item)
    else:
        yield value


def text_of(message):
    content = (message or {}).get("content", "")
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content or "")


def parse_tool_calls(text):
    for body in re.findall(r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL):
        match = re.search(r"<function=([^>]+)>", body)
        if not match:
            continue
        function = match.group(1)
        parameter = re.search(r"<parameter[>=]\s*", body)
        if not parameter:
            continue
        try:
            args, end = json.JSONDecoder().raw_decode(body, parameter.end())
        except json.JSONDecodeError:
            continue
        if not isinstance(args, dict) or not re.match(r"\s*</parameter>", body[end:]):
            continue
        yield function, args


async def run(base_url, prompt, request_kwargs, metadata, **_kwargs):
    # Dataset rows are conversations when dataset-side templating is disabled.
    conversation = copy.deepcopy(prompt) if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    messages = [{"role": "system", "content": SYSTEM}, *conversation]
    # Retain the actual conversation for TITO mismatch diagnosis. No HTTP headers.
    trajectory = {"messages": messages, "request_settings": {key: request_kwargs[key] for key in ("temperature", "top_p", "top_k", "max_tokens", "chat_template_kwargs") if key in request_kwargs}, "finish_reasons": []}
    actual_calls = []
    response = None
    for _turn in range(6):
        body = {**request_kwargs, "model": f"{DEFAULT_MODEL_NAME}:{LORA_ADAPTER_NAME}", "messages": messages, "max_tokens": request_kwargs.get("max_tokens", 768), "n": 1}
        completion = await post(f"{base_url}/v1/chat/completions", body)
        choice = completion["choices"][0]
        message = choice["message"]
        text = text_of(message)
        messages += [message]
        trajectory["finish_reasons"].append(choice.get("finish_reason"))
        if choice.get("finish_reason") == "length":
            response = text
            break
        calls = list(parse_tool_calls(text)) or [(item["function"]["name"], json.loads(item["function"]["arguments"])) for item in (message.get("tool_calls") or [])]
        if not calls:
            response = text
            break
        for function, args in calls:
            observation = {"error": "unknown_tool"}
            try:
                if function == "describe_schema":
                    observation = await asyncio.to_thread(describe_database, metadata["db_path"], args.get("table_name"))
                elif function == "run_sql":
                    observation = await asyncio.to_thread(execute_query, metadata["db_path"], args["sql"])
                actual_calls.append({"tool": function, "args": args, "tool_output": observation})
                messages += [{"role": "tool", "tool_call_id": str(len(actual_calls)), "name": function, "content": json.dumps(observation, default=str)}]
            except DatabaseUnavailableError:
                raise
            except Exception as exc:
                observation = {"error": type(exc).__name__, "message": str(exc)[:400]}
                actual_calls.append({"tool": function, "args": args, "tool_output": observation})
                messages += [{"role": "tool", "tool_call_id": str(len(actual_calls)), "name": function, "content": json.dumps(observation)}]
        if response is not None:
            break
    if response is None:
        return {
            **trajectory,
            "reward": 0.0,
            "execution_correct": False,
            "ungradeable": "max_turns",
            "calls": actual_calls,
            "num_turns": 6,
        }
    try:
        extracted = final_sql(response)
    except ValueError as exc:
        return {
            **trajectory,
            "reward": 0.0,
            "execution_correct": False,
            "format_error": f"invalid_final_sql: {exc}",
            "calls": actual_calls,
            "num_turns": _turn + 1,
            "response_length": len(response),
            "final_sql": response,
        }
    grading = await asyncio.to_thread(ex_score, extracted, metadata["gold_sql"], metadata["db_path"])
    return {
        **trajectory,
        "reward": float(grading["correct"]),
        "execution_correct": grading["correct"],
        "calls": actual_calls,
        "num_turns": _turn + 1,
        "response_length": len(response),
        "final_sql": response,
    }
