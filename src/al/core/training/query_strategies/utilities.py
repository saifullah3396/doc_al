from xai_torch.core.constants import DataKeys


def logits_output_transform(output: dict):
    return output[DataKeys.LOGITS]
