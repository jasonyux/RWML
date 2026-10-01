"""Text-only RL dataset for RWML training with verl.

Use with:
    data.custom_cls.path=rwml/dataset.py
    data.custom_cls.name=TextOnlyRLHFDataset
"""

from verl.utils.dataset.rl_dataset import RLHFDataset


class TextOnlyRLHFDataset(RLHFDataset):
    def _build_messages(self, example: dict):
        return example.pop(self.prompt_key)
