"""Shared checkpoint loading and preprocessing for the app and evaluation tools."""
from pathlib import Path
import json

import torch
from torch import nn
from torchvision import models, transforms


class BasicCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256 * 14 * 14, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.4), nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_basic_cnn(num_classes):
    return BasicCNN(num_classes)


def build_resnet50_transfer(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_efficientnet_b0_transfer(num_classes):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def load_inference_artifacts(model_path, class_map_path=None, device=None):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model_name = checkpoint.get("model_name") or checkpoint.get("best_params", {}).get("model_name")
    if model_name == "BasicCNN":
        model_name = "Basic CNN"
    mapping = checkpoint.get("idx_to_class", {})
    if not mapping and checkpoint.get("class_names"):
        mapping = dict(enumerate(checkpoint["class_names"]))
    if not mapping and class_map_path and Path(class_map_path).exists():
        mapping = json.loads(Path(class_map_path).read_text(encoding="utf-8")).get("idx_to_class", {})
    mapping = {int(k): str(v) for k, v in mapping.items()}
    if not mapping or sorted(mapping) != list(range(len(mapping))) or len(set(mapping.values())) != len(mapping):
        raise ValueError("Class mapping must contain unique labels at contiguous indices starting from zero.")
    img_size = int(checkpoint.get("img_size", 224))
    if img_size <= 0 or (model_name == "Basic CNN" and img_size != 224):
        raise ValueError(f"Unsupported image size {img_size} for {model_name}.")
    builders = {"Basic CNN": build_basic_cnn, "ResNet50": build_resnet50_transfer,
                "EfficientNet-B0": build_efficientnet_b0_transfer}
    if model_name not in builders:
        raise ValueError(f"Unsupported model_name in checkpoint: {model_name}")
    model = builders[model_name](len(mapping))
    state = checkpoint.get("state_dict", checkpoint.get("model_state_dict"))
    if state is None:
        raise ValueError("Checkpoint has no state_dict or model_state_dict.")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    # Existing checkpoints were trained with RGB [0,1], without ImageNet normalization.
    transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    return model, device, mapping, transform, model_name
