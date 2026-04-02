"""Shared model loading for FastAPI and Streamlit."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models, transforms

APP_DIR = Path(__file__).parent.resolve()
DATA_ROOT = APP_DIR / "data" / "recyclable-household-waste" / "images"
CONFIG_PATH = APP_DIR / "deployment_config.json"
CLASS_NAMES_JSON = APP_DIR / "class_names.json"


def load_deployment_config() -> dict:
    if not CONFIG_PATH.exists():
        return {"checkpoint": "best_waste_model.pth", "architecture": "resnet18"}
    with open(CONFIG_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {
        "checkpoint": data.get("checkpoint", "best_waste_model.pth"),
        "architecture": (data.get("architecture") or "resnet18").lower().strip(),
    }


def build_model_for_architecture(arch: str, num_classes: int) -> nn.Module:
    arch = arch.lower().strip()
    if arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch == "resnet34":
        m = models.resnet34(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)
        return m
    if arch == "densenet121":
        m = models.densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m
    if arch == "convnext_tiny":
        m = models.convnext_tiny(weights=None)
        in_features = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_features, num_classes)
        return m
    raise ValueError(
        f"Unknown architecture: {arch}. "
        "Use: resnet18, resnet34, efficientnet_b0, densenet121, convnext_tiny"
    )


def resolve_class_root(base_root: Path) -> Path:
    candidates = [base_root / "images", base_root]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Dataset class root not found under: {base_root}")


def load_class_names() -> List[str]:
    """Prefer bundled class_names.json (works on Streamlit Cloud without dataset)."""
    if CLASS_NAMES_JSON.exists():
        with open(CLASS_NAMES_JSON, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            return list(data)
        if isinstance(data, dict) and "classes" in data:
            return list(data["classes"])

    try:
        class_root = resolve_class_root(DATA_ROOT)
        classes = sorted([p.name for p in class_root.iterdir() if p.is_dir()])
        if classes:
            return classes
    except (FileNotFoundError, OSError):
        pass

    raise RuntimeError(
        f"No class labels found. Add {CLASS_NAMES_JSON.name} next to the app, "
        f"or place dataset class folders under {DATA_ROOT}."
    )


def get_preprocess():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_trained_model() -> Tuple[
    Optional[nn.Module],
    List[str],
    torch.device,
    str,
    Path,
    Optional[str],
]:
    """
    Returns: model, class_names, device, architecture, checkpoint_path, error (None if ok).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = load_class_names()
    cfg = load_deployment_config()
    checkpoint_path = APP_DIR / cfg["checkpoint"]
    architecture = cfg["architecture"]

    if not checkpoint_path.exists():
        return (
            None,
            class_names,
            device,
            architecture,
            checkpoint_path,
            (
                f"Checkpoint not found: {checkpoint_path}. "
                "Train in the notebook and save weights, or fix deployment_config.json."
            ),
        )

    try:
        model = build_model_for_architecture(architecture, len(class_names))
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model, class_names, device, architecture, checkpoint_path, None
    except Exception as exc:
        return (
            None,
            class_names,
            device,
            architecture,
            checkpoint_path,
            f"Failed to load checkpoint: {exc}",
        )


def predict_image(
    model: nn.Module,
    class_names: List[str],
    device: torch.device,
    preprocess,
    image,
):
    """image: PIL Image RGB. Returns predicted_class, confidence, top_k list."""
    x = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    top_probs, top_idx = torch.topk(probs, k=min(5, len(class_names)))
    predicted_idx = int(top_idx[0].item())
    predicted_class = class_names[predicted_idx]
    confidence = float(top_probs[0].item())

    top_k = [
        {
            "class_name": class_names[int(idx.item())],
            "confidence": float(prob.item()),
        }
        for prob, idx in zip(top_probs, top_idx)
    ]
    return predicted_class, confidence, top_k
