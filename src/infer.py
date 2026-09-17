"""Egitilmis checkpoint ile tek bir goruntu uzerinde cikarim + gorsellestirme.

Kullanim:
    python -m src.infer --checkpoint checkpoints/best.pth --image data/raw/train_images/0002cc93b.jpg
"""
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from .dataset import NUM_CLASSES
from .model import build_model
from .transforms import get_val_transforms

# Her defect sinifi (1-4) icin ayri bir renk -> overlay'de kariskiklik olmasin
DEFECT_COLORS = np.array([
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
], dtype=np.uint8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--threshold", type=float, default=0.5)  # sigmoid olasiligini binary maskeye cevirme esigi
    p.add_argument("--out", default=None, help="Overlay PNG cikti yolu (verilmezse gosterilir)")
    return p.parse_args()


def overlay_masks(image: np.ndarray, masks: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    # masks: (C, H, W) binary -> her kanali kendi rengiyle orijinal goruntunun uzerine harmanla
    overlay = image.copy()
    for c in range(masks.shape[0]):
        color_layer = np.zeros_like(image)
        color_layer[masks[c] > 0] = DEFECT_COLORS[c]
        overlay = cv2.addWeighted(overlay, 1.0, color_layer, alpha, 0)
    return overlay


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # checkpoint icindeki "args" ile egitimde kullanilan mimari/encoder otomatik yakalanir
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    model = build_model(ckpt_args.get("arch", "unet"), ckpt_args.get("encoder", "resnet34"))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # OpenCV BGR okur, model RGB bekler
    image_bgr = cv2.imread(args.image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # egitimdeki gibi CLAHE + normalize uygulanir; maske burada kullanilmiyor, sadece pipeline'i tutmak icin dummy
    transform = get_val_transforms(height=image_rgb.shape[0], width=image_rgb.shape[1])
    dummy_mask = np.zeros((*image_rgb.shape[:2], NUM_CLASSES), dtype=np.float32)
    augmented = transform(image=image_rgb, mask=dummy_mask)
    tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()  # multilabel -> her kanal bagimsiz olasilik
    masks = (probs > args.threshold).astype(np.uint8)

    # PadIfNeeded orijinal boyuttan buyutmus olabilir, gorsellestirme icin orijinal boyuta geri kirp
    h, w = image_rgb.shape[:2]
    masks = masks[:, :h, :w]
    overlay = overlay_masks(image_rgb, masks)

    if args.out:
        cv2.imwrite(args.out, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Kaydedildi: {args.out}")
    else:
        plt.figure(figsize=(16, 6))
        plt.imshow(overlay)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
