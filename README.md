# ECE 285 HW1: VAE on Chest X-rays

A convolutional VAE trained on the Kaggle chest X-ray dataset at 64 × 64, scored with Inception Score and FID from pytorch-ignite. The two-page NeurIPS-format report is `report/hw1_report.pdf`.

The code is three files:

- `src/data_process.py`: image preprocessing and `CachedImageDataset`.
- `src/model.py`: the VAE.
- `src/train.py`: training, checkpoints, loss curves, generated images, and IS/FID.

Install the dependencies with `python -m pip install -r requirements.txt`, and run every command from the repository root.

## Data

Download the [chest X-ray dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) and unzip it into `data/`, so that `data/chest_xray/train/` and `data/chest_xray/test/` exist. Then cache the images:

```bash
unzip chest-xray-pneumonia.zip -d data
python src/data_process.py
```

Only `train/` and `test/` are read; the 16-image `val/` split, the nested `chest_xray/chest_xray` copy, and `__MACOSX` are ignored. The two splits are cached separately as `data/processed/train.pt` and `test.pt`. Images are converted to grayscale and bilinearly resized to a 64 × 64 square, with no augmentation. The dataset returns float32 images in `[0, 1]` and no labels. Add `--overwrite` to rebuild the cache. `--image-size` changes the cached resolution, but `src/model.py` expects 64 × 64.

## Training

Defaults are 64 × 64 images, latent dimension 128, Adam with learning rate `1e-3`, batch size 128, 100 epochs, and seed 42. The loss is BCE + KL, each summed over one image and then averaged over the batch, with KL weight 1.

```bash
python src/train.py
```

Without `--output-dir`, each run creates a new `outputs/vae_YYYYMMDD_HHMMSS/`, so CSVs and checkpoints from different runs never mix. The run in the report used 1000 epochs:

```bash
python src/train.py --epochs 1000 --output-dir outputs/vae_latent128_e1000
```

Each output folder contains:

- `config.json`: the settings of the run.
- `history.csv`, `loss.png`: train and test total, reconstruction, and KL loss for every epoch.
- `checkpoints/epoch_XXX.pt`: model, optimizer, epoch, and config. One is saved every epoch at 35.5 MB each, so a 1000-epoch run writes about 35 GB.
- `samples/epoch_XXX.png`: every 10 epochs and at the last epoch, an 8 × 8 grid decoded from the same 64 standard-normal `z`.
- `reconstructions/epoch_XXX.png`: the same 32 test images in the top half and their reconstructions in the bottom half.
- `metrics.json`: IS/FID of the last checkpoint, the real-image control, the sampled indices, the seed, and package versions.

For a quick check, add `--epochs 1 --skip-eval`. The first evaluation downloads about 104 MB of Inception weights; use MPS or CUDA.

## IS and FID

IS and FID use the default Inception-v3 of pytorch-ignite, which outputs ImageNet 1000-class probabilities. Papers use 2048-dimensional features instead, so these FID values are much smaller and not comparable with published ones. Real images are a seeded random subset of 2,000 training images; fake images are decoded from the same number of standard-normal `z`. Before scoring, grayscale images are bilinearly resized to 299 × 299 and repeated to three channels, with values kept in `[0, 1]`. A disjoint set of 2,000 real images gives the real-vs-real FID and the real-image IS under the same setup. The report compares scores only within this setup.

| `epoch_1000.pt`, 2,000 images per set | FID ↓ | IS ↑ |
| --- | --- | --- |
| VAE samples vs. real | 0.078 | 1.883 |
| Real vs. real | 0.0011 | 2.206 |

## Pretrained checkpoint

`epoch_1000.pt` (35.5 MB) is the last epoch of the 1000-epoch run and the checkpoint scored in the report. It is attached to the [v1.0 release](https://github.com/Lukelaitw/2026Fall-UCSD-ECE285-HW1/releases/tag/v1.0) rather than stored in git:

```bash
curl -L --create-dirs -o outputs/vae_latent128_e1000/checkpoints/epoch_1000.pt \
  https://github.com/Lukelaitw/2026Fall-UCSD-ECE285-HW1/releases/download/v1.0/epoch_1000.pt
```

It holds `model_state_dict`, `optimizer_state_dict` (Adam), `epoch`, and `config`. To draw 64 samples:

```python
import sys
import torch
from torchvision.utils import save_image

sys.path.insert(0, "src")
from model import VAE

ckpt = torch.load("outputs/vae_latent128_e1000/checkpoints/epoch_1000.pt",
                  map_location="cpu", weights_only=True)
model = VAE(latent_dim=ckpt["config"]["latent_dim"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
with torch.no_grad():
    images = model.decode(torch.randn(64, ckpt["config"]["latent_dim"]))
save_image(images, "samples.png", nrow=8)
```
