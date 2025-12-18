import sys
from pathlib import Path
import torch
from torchvision.transforms import functional as TF
from PIL import Image


def load_image(image_path):
    """Load an image from a file path and convert it to a tensor."""
    image = Image.open(image_path).convert("RGB")
    return TF.to_tensor(image)


def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> float:
    """Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        x (torch.Tensor): The first image tensor.
        y (torch.Tensor): The second image tensor.
        data_range (float): The data range of the input images.

    Returns:
        float: The PSNR value in decibels (dB).
    """
    mse = torch.mean((x - y) ** 2)
    if mse == 0:
        return float("inf")
    psnr_value = 20 * torch.log10(data_range / torch.sqrt(mse))
    return psnr_value.item()


def main():
    if len(sys.argv) != 3:
        print("Usage: python psnr.py <image1_path> <image2_path>")
        sys.exit(1)

    image1_path = Path(sys.argv[1])
    image2_path = Path(sys.argv[2])

    if not image1_path.is_file() or not image2_path.is_file():
        print("Both arguments must be valid image file paths.")
        sys.exit(1)

    img1 = load_image(image1_path)
    img2 = load_image(image2_path)

    if img1.shape != img2.shape:
        print("Images must have the same dimensions.")
        sys.exit(1)

    psnr_value = psnr(img1, img2)
    print(f"PSNR between the two images: {psnr_value:.2f} dB")


if __name__ == "__main__":
    main()
