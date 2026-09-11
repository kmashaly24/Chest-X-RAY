"""Model definitions: ViT backbone + Asymmetric Loss."""
import torch
import torch.nn as nn
from torchvision import models


class ChestViT(nn.Module):
    """ViT-B/16 backbone with the last ``unfreeze_last`` encoder blocks (plus the
    final LayerNorm) trainable, and an MLP head for multi-label classification.

    Must match the architecture used to train the saved checkpoint exactly.

    Args:
        num_classes (int): Number of disease labels (output logits).
        unfreeze_last (int): How many trailing encoder blocks to fine-tune.
        head_dropout (float): Dropout probability inside the MLP head.
        pretrained (bool): Use ImageNet weights (training) or random init
            (inference, where the checkpoint fully overwrites the state dict).
    """

    def __init__(self, num_classes=14, unfreeze_last=4, head_dropout=0.3, pretrained=True):
        super().__init__()
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = models.vit_b_16(weights=weights)

        # 1) Freeze the whole backbone ...
        for p in self.backbone.parameters():
            p.requires_grad = False
        # 2) ... then unfreeze the last N encoder blocks ...
        for blk in self.backbone.encoder.layers[-unfreeze_last:]:
            for p in blk.parameters():
                p.requires_grad = True
        # 3) ... and the final encoder LayerNorm.
        for p in self.backbone.encoder.ln.parameters():
            p.requires_grad = True

        # 4) Replace the classifier head with an MLP (always trainable).
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(512, num_classes),
        )
        for p in self.backbone.heads.head.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for imbalanced multi-label classification."""

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        x_sig = torch.sigmoid(logits)
        xs_pos, xs_neg = x_sig, 1 - x_sig
        if self.clip and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        loss = (targets * torch.log(xs_pos.clamp(min=self.eps))
                + (1 - targets) * torch.log(xs_neg.clamp(min=self.eps)))

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = xs_pos * targets + xs_neg * (1 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            loss *= torch.pow(1 - pt, gamma)

        return -loss.mean()