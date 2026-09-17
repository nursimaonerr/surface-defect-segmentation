"""Batch bazinda Mean IoU ve Dice (F1) hesaplari, sinif bazinda kirilim dahil."""
import torch
import segmentation_models_pytorch as smp

from .dataset import NUM_CLASSES

CLASS_NAMES = [f"defect_{i+1}" for i in range(NUM_CLASSES)]


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict:
    probs = torch.sigmoid(logits)
    # tp/fp/fn/tn: (batch, num_classes) — her ornek ve sinif icin ayri ayri sayilir
    tp, fp, fn, tn = smp.metrics.get_stats(
        probs, targets.long(), mode="multilabel", threshold=threshold
    )

    # reduction="none" + batch ekseninde mean -> once ornek bazinda IoU/Dice, sonra batch ortalamasi (sinif basina)
    iou_per_class = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none").mean(dim=0)
    dice_per_class = smp.metrics.f1_score(tp, fp, fn, tn, reduction="none").mean(dim=0)  # F1 == Dice katsayisi

    metrics = {
        "mean_iou": iou_per_class.mean().item(),   # case'in ana metrigi: Mean IoU
        "mean_dice": dice_per_class.mean().item(),  # case'in ana metrigi: Dice Coefficient
    }
    for name, iou_c, dice_c in zip(CLASS_NAMES, iou_per_class, dice_per_class):
        metrics[f"iou_{name}"] = iou_c.item()
        metrics[f"dice_{name}"] = dice_c.item()
    return metrics
