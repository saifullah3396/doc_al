from typing import Callable


class JointBatchToTensorDataCollator:
    def __init__(self, wrapped_collator: Callable) -> None:
        self._wrapped_collator = wrapped_collator

    def __call__(self, features):
        features_labeled = [sample["labeled"] for sample in features]
        features_unlabeled = [sample["unlabeled"] for sample in features]
        features_labeled = self._wrapped_collator(features_labeled)
        features_unlabeled = self._wrapped_collator(features_unlabeled)
        return {"labeled": features_labeled, "unlabeled": features_unlabeled}
