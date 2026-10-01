import re


def alfworld_projection(actions: list[str], action_pools: list[list[str]]):
    """Extracts the action in the last <action>...</action> block of each response.

    An action is valid if it is admissible, the response has a <think> block, and it has no Chinese characters.
    """
    valids = [0] * len(actions)
    for i, response in enumerate(actions):
        lowered = response.lower()
        start, end = lowered.rfind("<action>"), lowered.rfind("</action>")
        if start == -1 or end == -1:
            actions[i] = lowered[-30:]
            continue
        if start >= end:
            actions[i] = "none"
        else:
            action = lowered[start + len("<action>") : end].strip()[:500]
            actions[i] = action
            valids[i] = int(action in action_pools[i])

        if response.find("<think>") == -1 or response.find("</think>") == -1:
            valids[i] = 0
        if re.search(r"[一-鿿]", response):
            valids[i] = 0
    return actions, valids
