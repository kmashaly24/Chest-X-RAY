"""Inference helpers shared by the Gradio app and CLI checks."""
import numpy as np
import torch

import config
from model import ChestViT


def load_artifacts(model_path, thresholds_path, device=None):
    """Load the saved model and per-class thresholds.

    The model is built with ``pretrained=False`` because the checkpoint already
    contains every parameter, so no ImageNet download is required.
    """
    device = device or config.get_device()
    model = ChestViT(num_classes=config.NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    thresholds = np.load(thresholds_path)
    return model, thresholds, device


def predict_image(image, model, thresholds, device):
    """Return {disease: probability} for every disease at/above its threshold."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = config.eval_transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
    probs = torch.sigmoid(logits).float().cpu().numpy()[0]

    results = {}
    for label, prob, threshold in zip(config.ALL_LABELS, probs, thresholds):
        if prob >= threshold:
            results[label] = round(float(prob), 4)
    return results