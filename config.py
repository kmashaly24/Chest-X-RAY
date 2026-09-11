"""Shared configuration + transforms for the Chest X-Ray project.

Label order and preprocessing are shared by training, inference and the Gradio
app so everything stays consistent with the saved checkpoint/thresholds.
"""
from torchvision import transforms


# NIH ChestX-ray14 pathologies in the fixed sorted order used at training time.
# Index i in this list corresponds to output neuron i of the model.
ALL_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]
NUM_CLASSES = len(ALL_LABELS)

# Normalization stats of the pre-trained ViT backbone (ImageNet).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

# Training-time augmentation (random, X-ray safe).
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.RandomAdjustSharpness(0.5, p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Deterministic evaluation pipeline (no randomness).
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")