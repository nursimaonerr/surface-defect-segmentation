"""Multilabel segmentasyon icin BCE + Dice birlesik kaybi."""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()  # piksel bazinda siniflandirma, sinif dengesizligine karsi Dice ile tamamlanir
        self.dice = smp.losses.DiceLoss(mode="multilabel", from_logits=True)  # kucuk/ince catlak bolgelerinde overlap'i dogrudan optimize eder

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + (1 - self.bce_weight) * self.dice(logits, targets)
