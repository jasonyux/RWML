import tiktoken

_TOKENIZER = tiktoken.get_encoding("cl100k_base")
MAX_ACTION_TOKENS = 512


def tau2bench_projection(actions: list[str]):
    """Turns policy responses into tau2-bench actions: a tool call (JSON) or a message to the user.

    Responses whose action is longer than MAX_ACTION_TOKENS are marked invalid and truncated.
    """
    valids = [1] * len(actions)
    for i, response in enumerate(actions):
        # drop the reasoning
        end = response.rfind("</think>")
        action = response[end + len("</think>") :].strip() if end != -1 else response
        # keep only the tool call, if any
        if "<tool_call>" in action:
            start, end = action.rfind("<tool_call>"), action.rfind("</tool_call>")
            action = action[start + len("<tool_call>") : end].strip()
        tokens = _TOKENIZER.encode(action, disallowed_special=())
        if len(tokens) > MAX_ACTION_TOKENS:
            action = _TOKENIZER.decode(tokens[-MAX_ACTION_TOKENS:])
            valids[i] = 0
        actions[i] = action
    return actions, valids
