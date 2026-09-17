"""Severstal train.csv icin run-length encode/decode yardimcilari."""
import numpy as np


def rle_decode(rle: str, shape: tuple[int, int]) -> np.ndarray:
    """rle: 'start length start length ...' (1-indexed, column-major). shape: (H, W)."""
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)  # once duz vektor olarak doldurulur
    if not isinstance(rle, str) or rle.strip() == "":
        return mask.reshape(shape, order="F")

    tokens = rle.split()
    starts = np.array(tokens[0::2], dtype=int) - 1  # Kaggle RLE 1-indexli, numpy 0-indexli
    lengths = np.array(tokens[1::2], dtype=int)
    for start, length in zip(starts, lengths):
        mask[start:start + length] = 1
    # order="F" (column-major): Severstal RLE piksel siralamasi satir degil sutun bazinda
    return mask.reshape(shape, order="F")


def rle_encode(mask: np.ndarray) -> str:
    """mask: (H, W) binary array -> 'start length ...' RLE string."""
    pixels = mask.flatten(order="F")  # decode ile ayni sutun-oncelikli sirada duzlestir
    pixels = np.concatenate([[0], pixels, [0]])  # baslangic/bitis sinirlarini yakalamak icin 0 ile sarmala
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1  # 0<->1 gecis noktalari = run baslangic/bitisleri
    runs[1::2] -= runs[::2]  # bitis indekslerini uzunluga cevir
    return " ".join(str(x) for x in runs)
