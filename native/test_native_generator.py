import argparse
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bird_evaluator import bird_agent
from bird_evaluator.rewards import reward_func
from miles.rollout.base_types import GenerateFnInput
from miles.rollout.generate_hub import agentic_tool_call
from miles.rollout.session.types import SessionRecord
from miles.utils.types import Sample


VALID_SQL = "SELECT value FROM items ORDER BY value"


class Tokenizer:
    @staticmethod
    def decode(token_ids):
        return " ".join(f"token-{token_id}" for token_id in token_ids)


class Tracer:
    session_id = "cpu-native-session"
    session_server_instance_id = None
    base_url = "http://session/sessions/cpu-native-session"

    async def collect_records(self):
        record = SessionRecord(
            timestamp=1.0,
            method="POST",
            path="/v1/chat/completions",
            request={"input_ids": [10, 11]},
            response={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "meta_info": {"output_token_logprobs": [[-0.25, 20]]},
                    }
                ]
            },
            status_code=200,
        )
        return [record], {"accumulated_token_ids": [10, 11, 20]}


def database():
    directory = tempfile.TemporaryDirectory(
        prefix="native-generator-", dir=Path(__file__).resolve().parent
    )
    path = Path(directory.name) / "tiny.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE items (value INTEGER)")
    connection.executemany("INSERT INTO items VALUES (?)", [(1,), (2,)])
    connection.commit()
    connection.close()
    return directory, path


def metadata(path):
    return {"db_path": str(path), "gold_sql": VALID_SQL}


class NativeGeneratorTest(unittest.IsolatedAsyncioTestCase):
    async def test_native_registration_invocation_and_reward_handoff(self):
        directory, path = database()
        self.addCleanup(directory.cleanup)
        parser = argparse.ArgumentParser()
        agentic_tool_call.generate.add_arguments(parser)
        args = parser.parse_args([
            "--custom-agent-function-path", "bird_evaluator.bird_agent.run",
        ])
        self.assertFalse(args.generate_multi_samples)
        args.session_server_ip = "session"
        args.session_server_port = 1
        requests = []
        agent_calls = []
        original_agent_run = bird_agent.run

        async def fake_post(url, body):
            requests.append((url, body))
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"```sql\n{VALID_SQL}\n```",
                            "tool_calls": None,
                        }
                    }
                ]
            }

        generation_input = GenerateFnInput(
            state=SimpleNamespace(args=args, tokenizer=Tokenizer()),
            sample=Sample(prompt="Return every value.", metadata=metadata(path)),
            sampling_params={"temperature": 1.0, "max_new_tokens": 768},
            evaluation=False,
        )

        async def recording_agent(**kwargs):
            agent_calls.append(kwargs)
            return await original_agent_run(**kwargs)

        with (
            patch.object(agentic_tool_call.OpenAIEndpointTracer, "create", AsyncMock(return_value=Tracer())),
            patch.object(bird_agent, "run", recording_agent),
            patch.object(bird_agent, "post", fake_post),
        ):
            output = await agentic_tool_call.generate(generation_input)
        sample = output.samples
        self.assertEqual(len(requests), 1)
        self.assertTrue(requests[0][0].endswith("/v1/chat/completions"))
        self.assertEqual(requests[0][1]["max_tokens"], 768)
        self.assertEqual(sample.tokens, [10, 11, 20])
        self.assertEqual(sample.loss_mask, [1])
        self.assertEqual(sample.metadata["execution_correct"], True)
        self.assertEqual(agent_calls[0]["metadata"]["session_server_id"], "session:1")
        self.assertEqual(agent_calls[0]["metadata"]["db_path"], str(path))
        self.assertEqual(await reward_func(args, sample), 1.0)
        print("PASS: actual native generate registration, invocation, token records, metadata and EX reward handoff")

    async def test_malformed_sql_is_reward_zero_not_infrastructure_failure(self):
        directory, path = database()
        self.addCleanup(directory.cleanup)
        responses = [
            "No SQL block here.",
            f"```sql\n{VALID_SQL}\n```\n```sql\n{VALID_SQL}\n```",
            "```sql\nDROP TABLE items\n```",
        ]
        for response in responses:
            async def fake_post(url, body, response=response):
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": response,
                                "tool_calls": None,
                            }
                        }
                    ]
                }

            with patch.object(bird_agent, "post", fake_post):
                result = await bird_agent.run("http://session", "question", {}, metadata(path))
            self.assertEqual(result["reward"], 0.0)
            self.assertFalse(result["execution_correct"])
            self.assertTrue(result["format_error"].startswith("invalid_final_sql:"))
            self.assertNotIn("error", result)
            self.assertEqual(await reward_func(None, SimpleNamespace(metadata=result)), 0.0)
        print("PASS: missing, multiple and non-readonly SQL answers return reward 0 with format diagnostics")

    async def test_runtime_errors_remain_distinct(self):
        directory, path = database()
        self.addCleanup(directory.cleanup)

        async def fake_post(url, body):
            raise RuntimeError("injected transport failure")

        with patch.object(bird_agent, "post", fake_post):
            with self.assertRaisesRegex(RuntimeError, "injected transport failure"):
                await bird_agent.run("http://session", "question", {}, metadata(path))


if __name__ == "__main__":
    unittest.main()
