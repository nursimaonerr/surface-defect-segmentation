"""Egitim betigi.

Kullanim:
    python -m src.train --arch unet --encoder resnet34 --epochs 30 --batch-size 8

Yerelde (CPU) hizli bir smoke-test icin --epochs 1 --limit-samples 16 kullanin;
gercek egitim Colab'da GPU ile yapilir (bkz. notebooks/train_colab.ipynb).
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from .dataset import SeverstalDataset, train_val_split
from .engine import train_one_epoch, validate
from .losses import BCEDiceLoss
from .model import build_model
from .transforms import get_train_transforms, get_val_transforms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--arch", default="unet", choices=["unet", "fpn"])
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--crop-height", type=int, default=256)
    p.add_argument("--crop-width", type=int, default=512)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--limit-samples", type=int, default=None, help="Hizli smoke-test icin veri sayisini sinirla")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--resume", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    csv_path = os.path.join(args.data_dir, "train.csv")
    images_dir = os.path.join(args.data_dir, "train_images")
    train_ids, val_ids = train_val_split(csv_path, val_size=args.val_size)
    if args.limit_samples:
        # CPU'da tum veri setiyle ugrasmadan pipeline'i dogrulamak icin (bkz. dosya basindaki smoke-test notu)
        train_ids = train_ids[:args.limit_samples]
        val_ids = val_ids[:max(1, args.limit_samples // 4)]

    train_ds = SeverstalDataset(
        csv_path, images_dir, image_ids=train_ids,
        transforms=get_train_transforms(args.crop_height, args.crop_width),
    )
    val_ds = SeverstalDataset(
        csv_path, images_dir, image_ids=val_ids,
        transforms=get_val_transforms(),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size // 2), shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.arch, args.encoder).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None  # sadece GPU'da mixed-precision

    start_epoch = 0
    best_iou = 0.0
    if args.resume and os.path.exists(args.resume):
        # Colab oturumu kesilirse kaldigi yerden devam edebilmek icin
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_iou = ckpt.get("best_iou", 0.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_stats = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"[epoch {epoch}] train_loss={train_stats['loss']:.4f} "
              f"val_loss={val_stats['loss']:.4f} "
              f"val_mean_iou={val_stats['mean_iou']:.4f} "
              f"val_mean_dice={val_stats['mean_dice']:.4f}")

        # her epoch sonunda "last" guncellenir, sadece iyilesme oldugunda "best" ayrica kaydedilir
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_iou": best_iou,
            "args": vars(args),  # infer.py checkpoint'ten mimari/encoder'i otomatik okuyabilsin diye
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, "last.pth"))
        if val_stats["mean_iou"] > best_iou:
            best_iou = val_stats["mean_iou"]
            ckpt["best_iou"] = best_iou
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "best.pth"))
            print(f"  -> yeni en iyi model kaydedildi (mean_iou={best_iou:.4f})")


if __name__ == "__main__":
    main()
