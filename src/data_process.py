"""Convert chest X-rays to cached grayscale tensors."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

HW1_DIR = Path(__file__).resolve().parents[1]


class CachedImageDataset(Dataset):
    """Return images only: float32, (1, H, W), values in [0, 1]."""

    def __init__(self, path):
        self.images = torch.load(path, map_location="cpu", weights_only=True)["images"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index].float().div_(255)


def prepare_split(data_root, output_dir, split, image_size=64, overwrite=False):
    destination = Path(output_dir) / f"{split}.pt"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists; use --overwrite to rebuild")
    paths = sorted(
        path for path in (Path(data_root) / split).glob("*/*")
        if path.is_file() and not path.name.startswith(".")
        and path.suffix.lower() in {".jpeg", ".jpg", ".png"}
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {Path(data_root) / split}")
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    images = torch.empty((len(paths), 1, image_size, image_size), dtype=torch.uint8)
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            image = source.convert("L").resize(
                (image_size, image_size), Image.Resampling.BILINEAR
            )
            images[index, 0] = torch.from_numpy(np.array(image, copy=True))
        if (index + 1) % 500 == 0 or index + 1 == len(paths):
            print(f"{split}: {index + 1}/{len(paths)}", flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images}, destination)
    print(f"Saved {destination}: {tuple(images.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=HW1_DIR / "data/chest_xray")
    parser.add_argument("--output-dir", type=Path, default=HW1_DIR / "data/processed")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for split in ("train", "test"):
        prepare_split(args.data_root, args.output_dir, split, args.image_size, args.overwrite)
