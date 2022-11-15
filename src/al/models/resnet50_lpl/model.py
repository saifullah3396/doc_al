""" Base Model class for the any model from Timm Repository. """


from dataclasses import dataclass, field

from torch import nn
from xai_torch.core.constants import DataKeys
from xai_torch.core.models.base_config import ModelConfig
from xai_torch.core.models.decorators import register_model
from xai_torch.core.models.image_base import BaseModuleForImageClassification
from xai_torch.core.models.utilities.general import load_model_from_online_repository
from xai_torch.core.training.constants import TrainingStage


class ResNet50LPL(nn.Module):
    def __init__(self, dim=224 * 224, pretrained=False, num_labels=10):
        super().__init__()
        from torchvision import models
        self.model = models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_labels)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x1 = self.model.layer1(x)
        x2 = self.model.layer2(x1)
        x3 = self.model.layer3(x2)
        x4 = self.model.layer4(x3)
        x = self.model.avgpool(x4)
        output = x.view(x.size(0), -1)
        output = self.model.fc(output)
        return output, [x1, x2, x3, x4]

    def get_embedding_dim(self):
        return self.dim


@register_model(reg_name="resnet50_lpl", task="image_classification")
class ResNet50LPLForImageClassification(BaseModuleForImageClassification):
    @dataclass
    class Config(ModelConfig):
        kwargs: dict = field(default_factory={})

    @property
    def model_name(self):
        return self.config.model_type

    def _build_model(self):
        import torch
        from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

        self.model = ResNet50LPL(
            num_labels=self.num_labels,
            pretrained=self.args.model_args.pretrained,
            # cache_dir=self.model_args.cache_dir,
        )
        self.loss_fn_eval = torch.nn.CrossEntropyLoss()

        # setup mixup function
        if self.training_args.cutmixup_args is not None:
            self.mixup_fn = self.training_args.cutmixup_args.get_fn(
                num_classes=self.num_labels, smoothing=self.training_args.smoothing
            )

        # setup loss accordingly if mixup, or label smoothing is required
        if self.mixup_fn is not None:
            # smoothing is handled with mixup label transform
            self.loss_fn_train = SoftTargetCrossEntropy(reduction="none")
        elif self.training_args.smoothing > 0.0:
            self.loss_fn_train = LabelSmoothingCrossEntropy(smoothing=self.training_args.smoothing, reduction="none")
        else:
            self.loss_fn_train = torch.nn.CrossEntropyLoss(reduction="none")
        self.loss_fn_eval = torch.nn.CrossEntropyLoss()

    def forward(self, image, label=None, stage=TrainingStage.predict):
        if stage in [TrainingStage.train, TrainingStage.test, TrainingStage.val] and label is None:
            raise ValueError(f"Label must be passed for stage={stage}")

        # apply mixup if required
        if stage == TrainingStage.train and self.mixup_fn is not None:
            image, label = self.mixup_fn(image, label)

        # compute logits
        logits, embeddings = self.model(image)
        if stage is TrainingStage.predict:
            if self.config.return_dict:
                return {
                    DataKeys.LOGITS: logits,
                    DataKeys.EMBEDDING: embeddings,
                }
            else:
                return logits, embeddings
        else:
            if stage is TrainingStage.train:
                loss = self.loss_fn_train(logits, label)
            else:
                loss = self.loss_fn_eval(logits, label)
            if self.config.return_dict:
                return {
                    DataKeys.LOSS: loss,
                    DataKeys.LOGITS: logits,
                    DataKeys.EMBEDDING: embeddings,
                    DataKeys.LABEL: label,
                }
            else:
                return (loss, logits, embeddings, label)
