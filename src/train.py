"""Train a chest X-ray VAE, save samples, and compute IS/FID."""

import argparse
import csv
import json
import math
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_process import CachedImageDataset, HW1_DIR
from model import VAE


def setup(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(device)


def log_metrics(path, epoch, **metrics):
    """Append scalar values computed by your own training code."""
    path = Path(path)
    fields = ["epoch", *metrics]
    values = [float(value) for value in metrics.values()]
    if not metrics or not all(math.isfinite(value) for value in values):
        raise ValueError("Provide finite numeric metrics")
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="") as stream:
            if next(csv.reader(stream)) != fields:
                raise ValueError("Metric columns must match the existing CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as stream:
        writer = csv.writer(stream)
        if not exists:
            writer.writerow(fields)
        writer.writerow([epoch, *values])


def save_checkpoint(path, model, optimizer, epoch, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch, "config": config}, stream)


def plot_history(csv_path, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Path(csv_path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        rows = list(reader)
    if not rows or "epoch" not in fields or len(fields) < 2:
        raise ValueError("CSV needs epoch and at least one metric column and row")
    groups = {}
    for field in fields:
        if field == "epoch":
            continue
        prefix, _, name = field.partition("_")
        key = name if prefix in {"train", "test", "val"} and name else field
        groups.setdefault(key, []).append(field)
    figure, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 3), squeeze=False)
    for axis, (name, columns) in zip(axes.flat, groups.items()):
        for column in columns:
            axis.plot([float(row["epoch"]) for row in rows],
                      [float(row[column]) for row in rows], label=column)
        axis.set(title=name, xlabel="Epoch")
        axis.legend()
        axis.grid(alpha=0.2)
    figure.tight_layout()
    output_path = Path(output_path)
    if output_path.resolve() == Path(csv_path).resolve():
        plt.close(figure)
        raise ValueError("Plot output must not overwrite the input CSV")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def vae_loss(x_hat, x, mu, logvar):
    batch_size = x.size(0)
    rec = torch.nn.functional.binary_cross_entropy(x_hat, x, reduction="sum") / batch_size
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    total = rec + kl
    return total, rec, kl


def run_loader(model, loader, device, optimizer=None):
    """Return mean total, reconstruction, and KL loss over every image."""
    totals = [0.0, 0.0, 0.0]
    seen = 0
    for images in loader:
        images = images.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        reconstruction, mu, logvar = model(images)
        loss, rec, kl = vae_loss(reconstruction, images, mu, logvar)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        batch = images.size(0)
        totals[0] += loss.item() * batch
        totals[1] += rec.item() * batch
        totals[2] += kl.item() * batch
        seen += batch
    return tuple(value / seen for value in totals)


def save_visuals(model, fixed_z, fixed_x, sample_path, recon_path):
    """Save an 8×8 sample grid and a fixed test-set reconstruction comparison."""
    samples = model.decode(fixed_z).detach().cpu()
    reconstruction, _, _ = model(fixed_x)
    reconstruction = reconstruction.detach().cpu()
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    recon_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(samples, sample_path, nrow=8)
    count = min(32, fixed_x.size(0))
    comparison = torch.cat([fixed_x[:count].detach().cpu(), reconstruction[:count]], dim=0)
    save_image(comparison, recon_path, nrow=8)


def to_inception(images):
    """Resize grayscale batches to the 3×299×299 inputs expected by Inception."""
    resized = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
    return resized.repeat(1, 3, 1, 1)


def stack_indices(dataset, indices):
    return torch.stack([dataset[int(index)] for index in indices])


def score_pairs(fake, real, device, batch_size):
    """FID between fake and real; Inception Score uses fake images only."""
    from ignite.metrics import FID, InceptionScore

    if fake.shape != real.shape:
        raise ValueError(f"fake and real shapes differ: {tuple(fake.shape)} vs {tuple(real.shape)}")
    fid = FID(device=device)
    inception_score = InceptionScore(device=device)
    with torch.no_grad():
        for start in range(0, real.size(0), batch_size):
            fake_batch = fake[start:start + batch_size].to(device)
            real_batch = real[start:start + batch_size].to(device)
            prepared_fake = to_inception(fake_batch)
            fid.update((prepared_fake, to_inception(real_batch)))
            inception_score.update(prepared_fake)
    return float(fid.compute()), float(inception_score.compute())


def generated_images(model, count, latent_dim, device, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    images = []
    model.eval()
    with torch.no_grad():
        remaining = count
        while remaining:
            batch = min(128, remaining)
            z = torch.randn(batch, latent_dim, generator=generator).to(device)
            images.append(model.decode(z).cpu())
            remaining -= batch
    return torch.cat(images, dim=0)


def evaluate(model, train_data, device, config, checkpoint_path, output_dir):
    n_images = config["eval_images"]
    if 2 * n_images > len(train_data):
        raise ValueError(
            f"Need two disjoint sets of {n_images} train images, but only {len(train_data)} are available"
        )
    sampler = torch.Generator()
    sampler.manual_seed(config["seed"])
    order = torch.randperm(len(train_data), generator=sampler)
    real_indices = order[:n_images]
    reference_indices = order[n_images:2 * n_images]
    real = stack_indices(train_data, real_indices)
    reference = stack_indices(train_data, reference_indices)
    if real.min() < 0 or real.max() > 1:
        raise ValueError("Evaluation images must stay in [0, 1]")

    fake = generated_images(
        model, n_images, config["latent_dim"], device, config["seed"]
    )
    print(f"Scoring {n_images} generated images against train split", flush=True)
    vae_fid, vae_is = score_pairs(fake, real, device, config["eval_batch_size"])
    print(f"Scoring {n_images} held-out real images against the same train split", flush=True)
    real_fid, real_is = score_pairs(reference, real, device, config["eval_batch_size"])

    results = {
        "checkpoint": str(checkpoint_path),
        "epoch": config["epochs"],
        "seed": config["seed"],
        "device": str(device),
        "real_split": "train",
        "n_images": n_images,
        "eval_batch_size": config["eval_batch_size"],
        "sampling": {
            "real_indices": "torch.randperm(len(train), generator seeded by seed); first n_images",
            "reference_indices": "the next n_images from that permutation; disjoint from real_indices",
            "fake": "z ~ Normal(0, I) from a CPU generator seeded by seed, then model.decode",
        },
        "real_indices": real_indices.tolist(),
        "reference_indices": reference_indices.tolist(),
        "preprocess": (
            "bilinear interpolate to 299x299, align_corners=False; "
            "repeat grayscale to 3 channels; values remain in [0, 1]"
        ),
        "feature_extractor": (
            "pytorch-ignite FID and InceptionScore defaults: "
            "ImageNet Inception-v3 1000-class probabilities (num_features=None)"
        ),
        "packages": {
            "torch": torch.__version__,
            "torchvision": version("torchvision"),
            "pytorch-ignite": version("pytorch-ignite"),
        },
        "vae": {"fid": vae_fid, "inception_score": vae_is},
        "real_vs_real": {"fid": real_fid, "inception_score": real_is},
    }
    destination = output_dir / "metrics.json"
    destination.write_text(json.dumps(results, indent=2) + "\n")
    print(f"VAE  FID {vae_fid:.6f}  IS {vae_is:.4f}", flush=True)
    print(f"Real FID {real_fid:.6f}  IS {real_is:.4f}", flush=True)
    print(f"Saved {destination}", flush=True)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-images", type=int, default=2000)
    parser.add_argument("--eval-batch-size", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = setup(args.seed)
    print(f"Device: {device}", flush=True)
    if device.type == "cpu":
        print("IS/FID on CPU is slow; MPS or CUDA is the expected device.", flush=True)

    config = {
        "latent_dim": args.latent_dim,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "image_size": 64,
        "kl_weight": 1.0,
        "eval_images": args.eval_images,
        "eval_batch_size": args.eval_batch_size,
    }
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = HW1_DIR / "outputs" / f"vae_{stamp}"
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else HW1_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Output: {output_dir}", flush=True)

    train_data = CachedImageDataset(HW1_DIR / "data/processed/train.pt")
    test_data = CachedImageDataset(HW1_DIR / "data/processed/test.pt")
    train_loader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    test_loader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=False)

    model = VAE(latent_dim=config["latent_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    fixed_z = torch.randn(64, config["latent_dim"], device=device)
    fixed_x = next(iter(test_loader)).to(device)
    history_path = output_dir / "history.csv"

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        train_loss, train_rec, train_kl = run_loader(model, train_loader, device, optimizer)

        model.eval()
        with torch.no_grad():
            test_loss, test_rec, test_kl = run_loader(model, test_loader, device)
            log_metrics(
                history_path,
                epoch,
                train_loss=train_loss,
                train_rec=train_rec,
                train_kl=train_kl,
                test_loss=test_loss,
                test_rec=test_rec,
                test_kl=test_kl,
            )
            checkpoint_path = output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"
            save_checkpoint(checkpoint_path, model, optimizer, epoch, config)
            if epoch % 10 == 0 or epoch == config["epochs"]:
                save_visuals(
                    model,
                    fixed_z,
                    fixed_x,
                    output_dir / "samples" / f"epoch_{epoch:03d}.png",
                    output_dir / "reconstructions" / f"epoch_{epoch:03d}.png",
                )

        print(
            f"epoch {epoch:03d}/{config['epochs']:03d}  "
            f"train loss {train_loss:.3f} rec {train_rec:.3f} kl {train_kl:.3f}  "
            f"test loss {test_loss:.3f} rec {test_rec:.3f} kl {test_kl:.3f}",
            flush=True,
        )

    plot_history(history_path, output_dir / "loss.png")
    chosen_checkpoint = output_dir / "checkpoints" / f"epoch_{config['epochs']:03d}.pt"
    print(f"Selected checkpoint: {chosen_checkpoint}", flush=True)
    if args.skip_eval:
        return

    loaded = torch.load(chosen_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(loaded["model_state_dict"])
    model.eval()
    evaluate(model, train_data, device, config, chosen_checkpoint, output_dir)


if __name__ == "__main__":
    main()
