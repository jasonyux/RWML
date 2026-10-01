ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
""".strip()

ALFWORLD_TEMPLATE_W_REFLECTION = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}

# Current Step
You are now at step {current_step} and your current observation is:
{current_observation}

Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first *briefly* reflect on the previous steps - summarize key information you've learned about the environment that is relevant to the task. Then, reason step-by-step to decide the best next action based on the current observation and the information from the previous steps. This reflection and reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
""".strip()
