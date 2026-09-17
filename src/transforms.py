"""Egitim/dogrulama transform pipeline'lari.

CLAHE ile kontrast normalizasyonu + degisken aydinlatma kosullarini simule eden
augmentation'lar (parlaklik/kontrast/gamma/golge) uygulanir.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(height: int = 256, width: int = 512) -> A.Compose:
    return A.Compose([
        A.RandomCrop(height=height, width=width),  # 1600px genisligi kucultup batch'i hizlandirir
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),  # lokal kontrasti esitler, dusuk isikta cizik/catlagi belirginlestirir
        A.OneOf([
            # farkli fabrika/gunisigi aydinlatmalarini simule eder -> modeli isik degisimine dayanikli kilar
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.RandomGamma(gamma_limit=(70, 130), p=1.0),
        ], p=0.7),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 2), p=0.3),  # yerel golge/parlama simulasyonu
        A.GaussNoise(std_range=(0.02, 0.08), p=0.2),  # sensor gurultusune karsi saglamlik
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # ImageNet on-egitimli encoder ile uyumlu normalizasyon
        ToTensorV2(),
    ])


def get_val_transforms(height: int = 256, width: int = 1600) -> A.Compose:
    # dogrulama/cikarimda rastgele augmentation yok; sadece boyut sabitleme + ayni kontrast normalizasyonu
    return A.Compose([
        A.PadIfNeeded(min_height=height, min_width=width, border_mode=0),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
