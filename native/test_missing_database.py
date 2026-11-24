import argparse
import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bird_evaluator import bird_agent
from bird_evaluator.bird_agent import run as run_agent
from bird_evaluator.rewards import reward_func
from bird_evaluator.sql_tools import DatabaseUnavailableError, describe_database, ex_score, execute_query
from miles.rollout.base_types import GenerateFnInput
from miles.rollout.generate_hub import agentic_tool_call
from miles.rollout.session.types import SessionRecord
from miles.utils.types import Sample


VALID_SQL = "SELECT value FROM items ORDER BY value"
INVALID_SQL = "SELECT absent_column FROM items"


class Tokenizer:
    @staticmethod
    def decode(token_ids):
        return " ".join(f"token-{token_id}" for token_id in token_ids)


class Tracer:
    session_id = "cpu-missing-db-session"
    session_server_instance_id = None
    base_url = "http://session/sessions/cpu-missing-db-session"

    async def collect_records(self):
        record = SessionRecord(
            timestamp=1.0,
            method="POST",
            path="/v1/chat/completions",
            request={"input_ids": [10, 11]},
            response={"choices": [{"finish_reason": "stop", "meta_info": {"output_token_logprobs": [[-0.25, 20]]}}]},
            status_code=200,
        )
        return [record], {"accumulated_token_ids": [10, 11, 20]}


class MissingDatabaseTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="missing-db-", dir=Path(__file__).resolve().parent)
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.database = root / "tiny.sqlite"
        self.missing = root / "absent.sqlite"
        connection = sqlite3.connect(self.database)
        connection.execute("CREATE TABLE items (value INTEGER)")
        connection.executemany("INSERT INTO items VALUES (?)", [(1,), (2,)])
        connection.commit()
        connection.close()
        self.before_digest = hashlib.sha256(self.database.read_bytes()).hexdigest()

    def metadata(self, path):
        return {"db_path": str(path), "gold_sql": VALID_SQL}

    def assert_unchanged_and_missing(self):
        self.assertFalse(self.missing.exists())
        self.assertEqual(hashlib.sha256(self.database.read_bytes()).hexdigest(), self.before_digest)

    async def agent_with_response(self, response):
        requests = []

        async def synthetic_post(url, body):
            requests.append((url, body))
            return {"choices": [{"message": {"role": "assistant", "content": response, "tool_calls": None}}]}

        with patch.object(bird_agent, "post", synthetic_post):
            result = await run_agent("http://session", "question", {}, self.metadata(self.missing))
        self.assertEqual(len(requests), 1)
        self.assertTrue(requests[0][0].endswith("/v1/chat/completions"))
        return result

    async def native_generate(self, path, predicted):
        parser = argparse.ArgumentParser()
        agentic_tool_call.generate.add_arguments(parser)
        args = parser.parse_args(["--custom-agent-function-path", "bird_evaluator.bird_agent.run"])
        self.assertFalse(args.generate_multi_samples)
        args.session_server_ip = "session"
        args.session_server_port = 1

        async def synthetic_post(url, body):
            return {"choices": [{"message": {"role": "assistant", "content": f"```sql\n{predicted}\n```", "tool_calls": None}}]}

        generation_input = GenerateFnInput(
            state=SimpleNamespace(args=args, tokenizer=Tokenizer()),
            sample=Sample(prompt="Return every value.", metadata=self.metadata(path)),
            sampling_params={"temperature": 1.0, "max_new_tokens": 768},
            evaluation=False,
        )
        with (
            patch.object(agentic_tool_call.OpenAIEndpointTracer, "create", AsyncMock(return_value=Tracer())),
            patch.object(bird_agent, "post", synthetic_post),
        ):
            output = await agentic_tool_call.generate(generation_input)
        return output.samples

    async def test_scorer_open_failures_are_typed_and_bad_sql_remains_false(self):
        self.assertTrue(ex_score(VALID_SQL, VALID_SQL, str(self.database))["correct"])
        invalid = ex_score(INVALID_SQL, VALID_SQL, str(self.database))
        self.assertIs(invalid["correct"], False)
        self.assertEqual(invalid["raw_result"]["res"], 0)
        for callback in (
            lambda: execute_query(str(self.missing), VALID_SQL),
            lambda: describe_database(str(self.missing)),
            lambda: ex_score("SELECT 1", "SELECT 1", str(self.missing)),
        ):
            with self.subTest(callback=callback):
                with self.assertRaises(DatabaseUnavailableError):
                    callback()
        self.assert_unchanged_and_missing()

    async def test_agent_propagates_open_failures_from_tools_and_final_grade(self):
        responses = [
            '<tool_call>\n<function=describe_schema>\n<parameter>{"table_name":"items"}</parameter>\n</function>\n</tool_call>',
            '<tool_call>\n<function=run_sql>\n<parameter>{"sql":"SELECT value FROM items"}</parameter>\n</function>\n</tool_call>',
            f"```sql\n{VALID_SQL}\n```",
        ]
        for response in responses:
            with self.subTest(response=response):
                with self.assertRaises(DatabaseUnavailableError):
                    await self.agent_with_response(response)
        self.assert_unchanged_and_missing()

    async def test_agent_correct_and_invalid_sql_on_readable_db(self):
        for predicted, expected in ((VALID_SQL, True), (INVALID_SQL, False)):
            with self.subTest(predicted=predicted, expected=expected):
                requests = []

                async def synthetic_post(url, body):
                    requests.append((url, body))
                    return {"choices": [{"message": {"role": "assistant", "content": f"```sql\n{predicted}\n```", "tool_calls": None}}]}

                with patch.object(bird_agent, "post", synthetic_post):
                    result = await run_agent("http://session", "question", {}, self.metadata(self.database))
                self.assertIs(result["execution_correct"], expected)
                self.assertEqual(result["reward"], float(expected))
                self.assertEqual(result["final_sql"], f"```sql\n{predicted}\n```")
                self.assertEqual(len(requests), 1)
        self.assert_unchanged_and_missing()

    async def test_native_generator_preserves_true_and_false_grades(self):
        for predicted, expected in ((VALID_SQL, True), (INVALID_SQL, False)):
            with self.subTest(predicted=predicted, expected=expected):
                sample = await self.native_generate(self.database, predicted)
                self.assertEqual(sample.tokens, [10, 11, 20])
                self.assertEqual(sample.loss_mask, [1])
                self.assertIs(sample.metadata["execution_correct"], expected)
                self.assertEqual(await reward_func(None, sample), float(expected))
        self.assert_unchanged_and_missing()

    async def test_native_generator_does_not_turn_open_failure_into_false(self):
        with self.assertLogs(agentic_tool_call.logger, level="WARNING") as captured:
            sample = await self.native_generate(self.missing, VALID_SQL)
        self.assertNotIn("execution_correct", sample.metadata)
        self.assertNotEqual(sample.metadata.get("execution_correct"), False)
        with self.assertRaisesRegex(RuntimeError, "Missing explicit SQL domain grade"):
            await reward_func(None, sample)
        self.assertIn("DatabaseUnavailableError", "\n".join(captured.output))
        self.assert_unchanged_and_missing()


if __name__ == "__main__":
    unittest.main()
