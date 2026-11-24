import asyncio
import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path

from bird_evaluator import bird_agent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


async def test_loop():
    with tempfile.TemporaryDirectory(prefix="agent-loop-", dir=Path(__file__).resolve().parent) as directory:
        database = Path(directory) / "tiny.sqlite"
        connection = sqlite3.connect(database)
        connection.execute("CREATE TABLE items (value INTEGER)")
        connection.executemany("INSERT INTO items VALUES (?)", [(1,), (2,)])
        connection.commit()
        connection.close()
        before = digest(database)
        requests = []
        responses = [
            '<tool_call>\n<function=describe_schema>\n<parameter>{"table_name":"items"}</parameter>\n</function>\n</tool_call>',
            '<tool_call>\n<function=run_sql>\n<parameter>{"sql":"SELECT value FROM items ORDER BY value"}</parameter>\n</function>\n</tool_call>',
            "```sql\nSELECT value FROM items ORDER BY value\n```",
        ]

        async def fake_post(url, body):
            requests.append((url, json.loads(json.dumps(body))))
            return {"choices": [{"message": {"role": "assistant", "content": responses[len(requests) - 1], "tool_calls": None}}]}

        original_post = bird_agent.post
        bird_agent.post = fake_post
        try:
            result = await bird_agent.run(
                "http://node:1/sessions/fixed",
                [{"type": "text", "text": "Return every value in ascending order."}],
                {"temperature": 1},
                {"gold_sql": "SELECT value FROM items ORDER BY value", "db_path": str(database)},
            )
        finally:
            bird_agent.post = original_post
        assert [call["tool"] for call in result["calls"]] == ["describe_schema", "run_sql"]
        assert all("db_path" not in call["args"] for call in result["calls"])
        assert result["execution_correct"] is True
        assert result["num_turns"] == 3
        assert len(requests) == 3
        assert all(url == "http://node:1/sessions/fixed/v1/chat/completions" for url, _ in requests)
        assert requests[1][1]["messages"][-1]["role"] == "tool"
        assert requests[1][1]["messages"][-1]["content"]
        assert requests[2][1]["messages"][-1]["role"] == "tool"
        assert all(body["n"] == 1 for _, body in requests)
        assert digest(database) == before
        print("PASS: describe_schema -> run_sql -> final SQL; metadata-bound DB, live tool observations, official EX, unchanged DB")


if __name__ == "__main__":
    asyncio.run(test_loop())
