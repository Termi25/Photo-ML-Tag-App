"""Neural network backbone + classification head for image tagging."""

import torch.nn as nn
from config import config_manager


class ImageClassificationModel(nn.Module):
    """ResNet / EfficientNet / MobileNet / VGG backbone with a configurable head.

    The backbone's original top layer is replaced with ``nn.Identity`` so the
    raw feature vector is passed to our own classification head.  Architecture
    is selected by ``config_manager.app_config.model_type``.
    """

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        cfg        = config_manager.hyperparameters
        model_type = config_manager.app_config.model_type

        if model_type == 'resnet18':
            from torchvision.models import resnet18, ResNet18_Weights
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            num_features  = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # type: ignore[assignment]

        elif model_type == 'resnet34':
            from torchvision.models import resnet34, ResNet34_Weights
            self.backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            num_features  = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # type: ignore[assignment]

        elif model_type == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            num_features  = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # type: ignore[assignment]

        elif model_type == 'efficientnet_b0':
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            num_features  = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()  # type: ignore[assignment]

        elif model_type == 'efficientnet_b1':
            from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
            self.backbone = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None)
            num_features  = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()  # type: ignore[assignment]

        elif model_type == 'mobilenet_v2':
            from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
            self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
            num_features  = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()  # type: ignore[assignment]

        elif model_type == 'vgg16':
            from torchvision.models import vgg16, VGG16_Weights
            self.backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
            num_features  = self.backbone.classifier[6].in_features
            self.backbone.classifier = nn.Identity()  # type: ignore[assignment]

        else:
            # Lightweight custom CNN fallback for unknown model type strings.
            print(f"Warning: Unknown model type '{model_type}', using custom CNN")
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            )
            num_features = 128

        # Classification head: optional hidden layers → output logits
        layers      = []
        in_features = int(num_features)  # type: ignore[arg-type]
        for hidden in (cfg.hidden_layers or []):
            layers += [nn.Linear(in_features, hidden), nn.ReLU(), nn.Dropout(cfg.dropout_rate)]
            in_features = hidden
        layers.append(nn.Linear(in_features, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def extract_features(self, x):
        """Return raw backbone features without the classification head."""
        return self.backbone(x)
