import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_rgb_float01(path: Path) -> np.ndarray:
    """HWC float32 in [0,1]."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def psnr_float01(ref: np.ndarray, img: np.ndarray) -> float:
    """
    ref, img: HWC float32 in [0,1], same shape
    """
    if ref.shape != img.shape:
        raise ValueError(f"Shape mismatch: ref {ref.shape} vs img {img.shape}")
    mse = np.mean((ref - img) ** 2, dtype=np.float64)
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)  # MAX=1 for [0,1]


def compute_folder_avg_psnr(
    size_dir: Path, suffix_ref="_input.png", suffix_img="_hidden.png"
) -> tuple[float, int]:
    ref_files = sorted(size_dir.glob(f"*{suffix_ref}"))
    total = 0.0
    count = 0

    for ref_path in tqdm(ref_files):
        base = ref_path.name[: -len(suffix_ref)]  # ID part
        img_path = size_dir / f"{base}{suffix_img}"
        if not img_path.exists():
            continue

        ref = load_rgb_float01(ref_path)
        img = load_rgb_float01(img_path)
        total += psnr_float01(ref, img)
        count += 1

    if count == 0:
        return float("nan"), 0
    return total / count, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", type=Path, help="root directory containing size_* subfolders"
    )
    parser.add_argument(
        "--ref_suffix", type=str, default="_input.png", help="reference file suffix"
    )
    parser.add_argument(
        "--img_suffix", type=str, default="_hidden.png", help="distorted file suffix"
    )
    args = parser.parse_args()

    size_dirs = sorted(
        [p for p in args.root.iterdir() if p.is_dir() and p.name.startswith("size_")]
    )
    if not size_dirs:
        raise RuntimeError(f"No size_* folders found under: {args.root}")

    for d in tqdm(size_dirs):
        avg, n = compute_folder_avg_psnr(
            d, suffix_ref=args.ref_suffix, suffix_img=args.img_suffix
        )
        print(f"{d.name}\tN={n}\tavg_PSNR={avg:.4f} dB")


if __name__ == "__main__":
    main()
