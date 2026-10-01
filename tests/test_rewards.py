"""Sanity checks for the RWML rewards, using a fake embedding server (run with `pytest tests`)."""

import types

import pytest

from rwml.rewards import alfworld, common, judge, tau2bench

THINK = "<think>The drawer is closed, so opening it should reveal its contents, which I have not seen before.</think>\n"


class FakeEmbeddings:
    """Embeds a text as its bag of characters, so identical texts have similarity 1."""

    def create(self, model, input, encoding_format):
        data = []
        for i, text in enumerate(input):
            text = text.split("Reference: ", 1)[-1]  # drop the query instruction
            data.append(types.SimpleNamespace(index=i, embedding=[text.count(c) + 1e-3 for c in "abcdefghijklmnopqrstuvwxyz0123456789"]))
        return types.SimpleNamespace(data=data)


@pytest.fixture(autouse=True)
def fake_embedding_server(monkeypatch):
    monkeypatch.setattr(common, "get_client", lambda api_base, api_key: types.SimpleNamespace(embeddings=FakeEmbeddings()))


EMBED_KWARGS = dict(embed_api_base="http://localhost:8000/v1")


def test_extract_next_state():
    assert common.extract_next_state("<think>x</think><next_state> a </next_state>") == "a"
    assert common.extract_next_state("<next_state>a</next_state><next_state>b</next_state>") == "b"
    assert common.extract_next_state("<next_state>a") == "none"
    assert common.extract_next_state("<next_state>" + "word " * 50 + "</next_state>", max_tokens=3) == "word word word"


def test_task_status_matches():
    assert common.task_status_matches("You see a mug.\nTask is not yet completed.", "Task is not yet completed.")
    assert not common.task_status_matches("Task is not yet completed or failed.", "Task is not yet completed.")
    assert not common.task_status_matches("Task is successfully completed!", "Task is failed.")
    assert not common.task_status_matches("", "Task is failed.")


def test_alfworld_reward():
    gt = "You open the drawer 1. The drawer 1 is open. In it, you see a keychain 2. Task is not yet completed."
    assert alfworld.compute_score("alfworld", THINK + f"<next_state>{gt}</next_state>", gt, {}, **EMBED_KWARGS) == 1.0
    assert alfworld.compute_score("alfworld", THINK + "<next_state>Nothing happens.</next_state>", gt, {}, **EMBED_KWARGS) == 0.0
    assert alfworld.compute_score("alfworld", THINK + gt, gt, {}, **EMBED_KWARGS) == 0.0  # no <next_state> tags


def test_tau2bench_reward():
    tool = {"metadata": {"response_type": "tool"}}
    user = {"metadata": {"response_type": "user"}}
    tool_gt = '{"type": "object", "properties": {"customer_id": {"type": "string"}}}\nTask is not yet completed.'
    user_gt = "Sure, my phone number is 555-123-2002.\nTask is not yet completed."

    def score(response, gt, extra_info):
        return tau2bench.compute_score("tau2bench", response, gt, extra_info, **EMBED_KWARGS)

    assert score(THINK + f"<next_state>{tool_gt}</next_state>", tool_gt, tool) == pytest.approx(1.0)
    assert score(THINK + f"<next_state>{user_gt}</next_state>", user_gt, user) == pytest.approx(1.0)
    # rouge is rounded to multiples of 0.2
    partial = '{"type": "object"}\nTask is not yet completed.'
    assert round(score(THINK + f"<next_state>{partial}</next_state>", tool_gt, tool), 6) in {0.0, 0.2, 0.4, 0.6, 0.8}
    # penalties: missing thinking (-0.1) and wrong task status (-0.5)
    assert score(f"<next_state>{user_gt}</next_state>", user_gt, user) == pytest.approx(0.9)
    wrong_status = user_gt.replace("not yet completed.", "successfully completed!")
    assert score(THINK + f"<next_state>{wrong_status}</next_state>", user_gt, user) == pytest.approx(0.5)
    # chat templates that open <think> in the prompt
    response = THINK.removeprefix("<think>") + f"<next_state>{user_gt}</next_state>"
    assert tau2bench.compute_score("tau2bench", response, user_gt, user, think_in_prompt=True, **EMBED_KWARGS) == pytest.approx(1.0)


def test_batched_preserves_order():
    gts = [f"You arrive at shelf {i}. Task is not yet completed." for i in range(1, 6)]
    responses = [THINK + f"<next_state>{gt}</next_state>" if i % 2 == 0 else "no prediction" for i, gt in enumerate(gts)]
    scores = alfworld.compute_score_batched(["alfworld"] * 5, responses, gts, [{}] * 5, num_workers=3, **EMBED_KWARGS)
    assert scores == [1.0, 0.0, 1.0, 0.0, 1.0]


def test_judge_reward(monkeypatch):
    extra_info = {"obs_text": "You are in the middle of a room.", "action_text": "<action>go to desk 1</action>"}
    gt = "You arrive at desk 1. On the desk 1, you see a mug 1. Task is not yet completed."

    def score(response, judge_output, **kwargs):
        message = types.SimpleNamespace(content=judge_output)
        create = lambda **_: types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])
        client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
        monkeypatch.setattr(judge, "get_client", lambda api_base, api_key: client)
        return judge.compute_score("alfworld", response, gt, extra_info, judge_api_base="http://localhost:8001/v1", **kwargs)

    response = THINK + f"<next_state>{gt}</next_state>"
    assert score(response, '<json>{"analysis": "ok", "score": 0.7}</json>') == pytest.approx(0.7)
    assert score(response, '{"score": 1.5}', binary=True) == pytest.approx(1.0)  # clipped to [0, 1]
    assert score(response, "not json") == 0.0
    assert score(f"<next_state>{gt}</next_state>", '{"score": 1.0}') == pytest.approx(0.9)  # no thinking
