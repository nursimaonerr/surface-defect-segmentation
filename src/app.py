"""Egitilmis modeli tarayicida test etmek icin basit Gradio arayuzu.

Kullanim:
    python -m src.app --checkpoint checkpoints/best.pth
"""
import argparse

import gradio as gr
import numpy as np
import torch

from .infer import load_model, predict_overlay

CLASS_LEGEND = "defect 1 (kirmizi) · defect 2 (yesil) · defect 3 (mavi) · defect 4 (sari)"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best.pth")
    p.add_argument("--share", action="store_true", help="Gradio'nun genel (public) linkini de olustur")
    return p.parse_args()


def build_demo(model, device: torch.device):
    def predict(image: np.ndarray, threshold: float):
        if image is None:
            return None
        return predict_overlay(model, image, device, threshold)

    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(label="Yuzey goruntusu", type="numpy"),
            gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Esik (threshold)"),
        ],
        outputs=gr.Image(label="Tahmin (overlay)"),
        title="CASE #04 — Yuzey Kusuru Segmentasyonu",
        description=f"U-Net (ResNet34) ile piksel bazinda kusur segmentasyonu. Renk lejandi: {CLASS_LEGEND}",
    )
    return demo


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    demo = build_demo(model, device)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
