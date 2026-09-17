"""Egitim/dogrulama donguleri."""
from collections import defaultdict

import torch
from tqdm import tqdm

from .metrics import compute_metrics


class AverageMeter:
    def __init__(self):
        self.sums = defaultdict(float)
        self.count = 0

    def update(self, values: dict, n: int = 1):
        for k, v in values.items():
            self.sums[k] += v * n
        self.count += n

    def avg(self) -> dict:
        return {k: v / self.count for k, v in self.sums.items()}


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None) -> dict:
    model.train()
    meter = AverageMeter()
    pbar = tqdm(loader, desc="train")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            # scaler sadece CUDA'da verilir (bkz. train.py) -> Colab GPU'da mixed-precision ile hizlanma
            with torch.autocast(device_type=device.type, enabled=True):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # CPU yolu (yerel gelistirme): autocast/scaler yok, standart fp32 egitim
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        meter.update({"loss": loss.item()}, n=images.size(0))
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return meter.avg()


@torch.no_grad()
def validate(model, loader, criterion, device) -> dict:
    model.eval()
    meter = AverageMeter()
    for images, masks in tqdm(loader, desc="val"):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)

        # IoU/Dice sadece validasyonda hesaplanir (egitimde hiz icin sadece loss izlenir)
        batch_metrics = compute_metrics(logits, masks)
        batch_metrics["loss"] = loss.item()
        meter.update(batch_metrics, n=images.size(0))
    return meter.avg()
