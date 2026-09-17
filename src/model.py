"""Model fabrikasi: U-Net veya FPN, ResNet34 encoder (ImageNet on segmentation_models_pytorch)."""
import segmentation_models_pytorch as smp

from .dataset import NUM_CLASSES


def build_model(arch: str = "unet", encoder: str = "resnet34", num_classes: int = NUM_CLASSES):
    arch = arch.lower()
    common_kwargs = dict(
        encoder_name=encoder,
        encoder_weights="imagenet",  # transfer learning: encoder ImageNet agirliklariyla baslar
        in_channels=3,
        classes=num_classes,  # 4 kanal = multilabel sigmoid cikisi (her defect sinifi bagimsiz)
    )
    if arch == "unet":
        return smp.Unet(**common_kwargs)
    if arch == "fpn":
        return smp.FPN(**common_kwargs)  # case dokumanindaki alternatif mimari
    raise ValueError(f"Bilinmeyen mimari: {arch} (unet | fpn)")
