# CASE #04 — Değişken Işık Şartlarına Dayanıklı Yüzey Segmentasyonu

Fabrikadaki gün ışığı değişimleri veya ortam aydınlatması dalgalanmalarına rağmen metal/plastik yüzeylerdeki kılcal çatlak ve çizikleri piksel bazında segmente eden sistem.

- **Veri Seti**: Severstal Steel Defect (Kaggle)
- **Model Mimarisi**: U-Net (ResNet34 Backbone), FPN alternatifi
- **Ana Metrikler**: Mean IoU, Dice Coefficient
- **Kilit Teknikler**: CLAHE ön işleme, Albumentations ile illumination augmentation

## Yapı

```
data/            veri (indirilir, git'e girmez)
src/              dataset, transforms, model, metrics, train, infer modülleri
notebooks/        Colab GPU eğitim notebook'u
scripts/          veri indirme betiği
```

- `src/rle.py` — Severstal `train.csv` için RLE encode/decode
- `src/dataset.py` — `SeverstalDataset` (4 sınıflı multilabel maske üretimi)
- `src/transforms.py` — CLAHE + illumination augmentation (Albumentations)
- `src/model.py` — U-Net / FPN (ResNet34 encoder, `segmentation_models_pytorch`)
- `src/losses.py` — BCE + Dice birleşik kaybı
- `src/metrics.py` — sınıf bazında ve ortalama IoU / Dice
- `src/engine.py` — eğitim/doğrulama döngüleri
- `src/train.py`, `src/infer.py` — CLI eğitim ve çıkarım betikleri

## Geliştirme ortamı

Windows üzerinde, yerel GPU olmadığı için veri hazırlığı/geliştirme yerelde (CPU), gerçek eğitim Google Colab'da (GPU) yapılır.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Kullanım

```powershell
# 1. Kaggle API token'ı %USERPROFILE%\.kaggle\kaggle.json içine yerleştirin, yarışma kurallarını kabul edin, sonra:
.\scripts\download_data.ps1

# 2. Hızlı yerel smoke-test (CPU, birkaç örnekle)
python -m src.train --epochs 1 --batch-size 2 --limit-samples 16

# 3. Gerçek eğitim: notebooks/train_colab.ipynb içinde Colab GPU üzerinde çalıştırın

# 4. Çıkarım / görselleştirme
python -m src.infer --checkpoint checkpoints/best.pth --image data/raw/train_images/<id>.jpg --out overlay.png
```

## Durum (2026-09-17)

**Tamamlanan:**
- [x] Proje iskeleti ve tüm `src/` modülleri yazıldı (rle, dataset, transforms, model, losses, metrics, engine, train, infer) — kodların yanına açıklayıcı yorumlar eklendi.
- [x] `scripts/download_data.ps1` (Windows) ve `scripts/download_data.sh` (Colab/Linux) veri indirme betikleri.
- [x] `notebooks/train_colab.ipynb` — Colab GPU eğitim akışı (repo klonlama → kaggle.json yükleme → indirme → `src.train` → Drive'a checkpoint kaydetme → örnek çıkarım).
- [x] Geliştirme ortamı Windows'a taşındı: native `.venv` (Python 3.12) kuruldu, `torch`/`torchvision`/`opencv`/`albumentations`/`segmentation-models-pytorch`/`kaggle` bağımlılıkları kuruldu (WSL'deki eski venv silindi, WSL ortamı bozuk çıktı).
- [x] Dummy veriyle uçtan uca smoke test geçti: transform pipeline → model forward-pass (U-Net/ResNet34) → BCE+Dice loss → IoU/Dice metrik hesaplama, hepsi Windows CPU'da çalışıyor.
- [x] RLE encode/decode roundtrip testi ayrı doğrulandı.

- [x] Disk alanı temizlendi (WSL'in bozuk Ubuntu dağıtımı + tekrarlayan dosyalar silindi) — kurulum sonrası ~1.3 GB'tan ~24 GB'a çıktı.
- [x] Kaggle API kimlik bilgisi (`access_token`) oluşturuldu, telefon doğrulaması yapıldı, yarışma kuralları kabul edildi.
- [x] Gerçek Severstal veri seti indirildi: `data/raw/train.csv` + 12.568 `train_images` + `test_images`.
- [x] Gerçek veriyle CPU smoke-test çalıştırıldı (`--epochs 1 --limit-samples 16`) — uçtan uca sorunsuz, `checkpoints/best.pth` ve `last.pth` üretildi.

**Bekleyen / sıradaki adımlar:**
- [ ] Proje henüz bir git deposu değil — Colab notebook'undaki `REPO_URL` alanı doldurulmadı; GitHub'a pushlanacaksa `git init` + remote ekleme gerekiyor (ya da Drive üzerinden dosya taşıma alternatifi).
- [ ] Gerçek/tam eğitim Colab GPU'da henüz çalıştırılmadı (yerelde sadece CPU smoke-test yapıldı, tüm veri setiyle tam eğitim yapılmadı).
- [x] `src.infer` gerçek smoke-test checkpoint'iyle (`checkpoints/best.pth`) denendi, `overlay_test.png` üretildi.

## Bilinen sorun: Windows'ta mutlak yol + Türkçe karakter

Proje `...\Masaüstü\...` altında olduğu için PowerShell'den bir betiğe **mutlak** dosya yolu (`C:\Users\...\Masaüstü\...`) argüman olarak verilirse, "ü" karakteri komut satırına aktarılırken bozuluyor ve OpenCV `can't open/read file` hatası veriyor. Çözüm: proje kök dizinindeyken **göreli yol** kullanın (`--image data/raw/train_images/<id>.jpg` gibi), mutlak yol vermeyin.
