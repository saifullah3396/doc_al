""" Base Model class for the any model from Timm Repository. """


from dataclasses import dataclass, field

from torch import nn
from xai_torch.core.constants import DataKeys
from xai_torch.core.models.base_config import ModelConfig
from xai_torch.core.models.decorators import register_model
from xai_torch.core.models.image_base import BaseModuleForImageClassification
from xai_torch.core.models.utilities.general import load_model_from_online_repository
from xai_torch.core.training.constants import TrainingStage
from typing import Dict, Any



@register_model(reg_name="resnet50_waal", task="image_classification")
class ResNet50WAALForImageClassification(BaseModuleForImageClassification):
    @dataclass
    class Config(ModelConfig):
        kwargs: dict = field(default_factory={})

    @property
    def model_name(self):
        return self.config.model_type

    @classmethod
    def load_model(
        cls,
        model_name,
        num_labels=None,
        use_timm=True,
        pretrained=True,
        config=None,
        **kwargs,
    ):
        import torch.nn.functional as F
        from torch import nn
        from torchvision import models
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_labels)
        return model

    def forward(self, x, step="features"):
        if step == "features":
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            return x.view(x.size(0), -1)
        else:
            return self.model.fc(x)

    def get_embedding_dim(self):
        return self.model.fc.in_features
