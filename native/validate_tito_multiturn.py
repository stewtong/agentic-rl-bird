import sys

if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11 or newer is required (found {sys.version.split()[0]})")

from miles.rollout.session.linear_trajectory import LinearTrajectory
from miles.utils.chat_template_utils.template import apply_chat_template
from miles.utils.chat_template_utils.tito_tokenizer import Qwen35TITOTokenizer
from miles.utils.processing_utils import load_tokenizer


def main(model_root, fixed_template):
    tokenizer = load_tokenizer(model_root, chat_template_path=fixed_template, trust_remote_code=True)
    tito = Qwen35TITOTokenizer(tokenizer, chat_template_kwargs={"enable_thinking": False})
    history = [
        {"role": "system", "content": "Use the SQL tool."},
        {"role": "user", "content": [{"type": "text", "text": "How many rows?"}]},
        {"role": "assistant", "content": "<tool_call>\n<function=run_sql>\n<parameter>{\"sql\":\"SELECT 1\"}</parameter>\n</function>\n</tool_call>"},
    ]
    following = history + [{"role": "tool", "name": "run_sql", "tool_call_id": "1", "content": '{"rows": [[1]]}'}]
    prompt_ids = tito.render_messages(history[:-1], add_generation_prompt=True, tokenize=True)
    rendered_history_ids = tito.render_messages(history, add_generation_prompt=False, tokenize=True)
    assert rendered_history_ids[: len(prompt_ids)] == prompt_ids
    assistant_ids = rendered_history_ids[len(prompt_ids) :]
    assert assistant_ids
    trajectory = LinearTrajectory()
    trajectory.update_pretokenized_state(history[:-1], history[-1], prompt_ids, assistant_ids, tito.max_trim_tokens)
    merged_ids = trajectory.prepare_pretokenized(following, tito_tokenizer=tito)
    canonical_ids = apply_chat_template(following, tokenizer=tokenizer, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    assert merged_ids == canonical_ids, (len(merged_ids), len(canonical_ids))
    assert following[-1]["role"] == "tool"
    print(f"Installed-session prefix + rendered assistant: exact {len(merged_ids)}-token tool continuation")


if __name__ == "__main__":
    main(*sys.argv[1:])
