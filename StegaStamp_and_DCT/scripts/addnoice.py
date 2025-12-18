import argparse
from pathlib import Path
import io
import torch
from torchvision.transforms import functional as TF
from PIL import Image
import os


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return TF.to_tensor(image)


def save_image(x, path):
    x = x.clamp(0.0, 1.0)
    img = TF.to_pil_image(x)
    img.save(path)


def rotate(x: torch.Tensor, degrees: float) -> torch.Tensor:
    return TF.rotate(x, angle=degrees, expand=False)


def gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    sigma is standard deviation in [0,1] scale
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    return (x + torch.randn_like(x) * sigma).clamp(0.0, 1.0)


def gaussian_blur(x: torch.Tensor, radius: float) -> torch.Tensor:
    from PIL import ImageFilter

    if radius < 0:
        raise ValueError("radius must be non-negative")
    x = x.clamp(0.0, 1.0)
    img = TF.to_pil_image(x)
    img_blur = img.filter(ImageFilter.GaussianBlur(radius))
    return TF.to_tensor(img_blur).clamp(0.0, 1.0)


def brightness(x: torch.Tensor, factor: float) -> torch.Tensor:
    # factor > 1 brighter, factor < 1 darker
    return (x * factor).clamp(0.0, 1.0)


def contrast(x: torch.Tensor, factor: float) -> torch.Tensor:
    # factor > 1 higher contrast, factor < 1 lower contrast
    mean = x.mean(dim=(1, 2), keepdim=True)
    return ((x - mean) * factor + mean).clamp(0.0, 1.0)


def jpeg_compress(x: torch.Tensor, quality: int) -> torch.Tensor:
    """
    Encode+decode JPEG in memory to create compression artifacts.
    quality: 1..95 (lower = stronger artifacts)
    """
    if not (1 <= quality <= 95):
        raise ValueError("quality must be in [1, 95]")
    img = TF.to_pil_image(x.clamp(0.0, 1.0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=False)
    buf.seek(0)
    img_jpeg = Image.open(buf).convert("RGB")
    return TF.to_tensor(img_jpeg).clamp(0.0, 1.0)


def main(args):
    images = os.listdir(args.input_dir)
    if args.gaussian:
        root_output_dir = os.path.join(args.output_dir, "gaussian")
        os.makedirs(root_output_dir, exist_ok=True)
        for i in range(1, 11):  # 1..10
            sigma = i * 0.05
            output_dir = os.path.join(root_output_dir, f"sigma_{sigma:.2f}")
            os.makedirs(output_dir, exist_ok=True)
            for image_name in images:
                image_path = args.input_dir / image_name
                x = load_image(image_path)
                x_noisy = gaussian_noise(x, sigma)
                save_image(x_noisy, os.path.join(output_dir, image_name))

    if args.rotate:
        root_output_dir = os.path.join(args.output_dir, "rotate")
        os.makedirs(root_output_dir, exist_ok=True)
        for angle in range(1, 11, 1):  # 0, 15, ..., 345
            output_dir = os.path.join(root_output_dir, f"angle_{angle}")
            os.makedirs(output_dir, exist_ok=True)
            for image_name in images:
                image_path = args.input_dir / image_name
                x = load_image(image_path)
                x_rotated = rotate(x, angle)
                save_image(x_rotated, os.path.join(output_dir, image_name))

    if args.blur:
        # radii in pixels; increase for stronger blur
        levels = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        root_output_dir = os.path.join(args.output_dir, "blur")
        os.makedirs(root_output_dir, exist_ok=True)
        for idx, r in enumerate(levels, start=1):
            output_dir = os.path.join(root_output_dir, f"level_{idx}_radius_{r:.2f}")
            os.makedirs(output_dir, exist_ok=True)
            for image_name in images:
                image_path = args.input_dir / image_name
                x = load_image(image_path)
                y = gaussian_blur(x, r)
                save_image(y, Path(output_dir) / image_name)

    if args.brightness:
        levels = [0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4]
        root_output_dir = os.path.join(args.output_dir, "brightness")
        os.makedirs(root_output_dir, exist_ok=True)
        for idx, f in enumerate(levels, start=1):
            output_dir = os.path.join(root_output_dir, f"level_{idx}_factor_{f:.2f}")
            os.makedirs(output_dir, exist_ok=True)
            for image_name in images:
                image_path = args.input_dir / image_name
                x = load_image(image_path)
                y = brightness(x, f)
                save_image(y, Path(output_dir) / image_name)

    if args.contrast:
        levels = [0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4]
        root_output_dir = os.path.join(args.output_dir, "contrast")
        os.makedirs(root_output_dir, exist_ok=True)
        for idx, f in enumerate(levels, start=1):
            output_dir = os.path.join(root_output_dir, f"level_{idx}_factor_{f:.2f}")
            os.makedirs(output_dir, exist_ok=True)
            for image_name in images:
                image_path = args.input_dir / image_name
                x = load_image(image_path)
                y = contrast(x, f)
                save_image(y, Path(output_dir) / image_name)

    if args.jpeg:
        levels = [
            90,
            80,
            70,
            60,
            50,
            40,
            30,
            20,
            10,
        ]  # lower quality = stronger artifacts
        root_output_dir = os.path.join(args.output_dir, "jpeg")
        os.makedirs(root_output_dir, exist_ok=True)
        for idx, q in enumerate(levels, start=1):
            output_dir = os.path.join(root_output_dir, f"level_{idx}_quality_{q}")
            os.makedirs(output_dir, exist_ok=True)
            for image_name in images:
                image_path = args.input_dir / image_name
                x = load_image(image_path)
                y = jpeg_compress(x, q)
                # Save as PNG to avoid applying JPEG twice.
                out_name = Path(image_name).with_suffix(".png").name
                save_image(y, Path(output_dir) / out_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=Path, required=True, help="Path to input image"
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True, help="Path to save output image"
    )
    parser.add_argument("--gaussian", action="store_true", help="Apply Gaussian noise")
    parser.add_argument("--rotate", action="store_true", help="Apply rotation")
    parser.add_argument("--blur", action="store_true", help="Apply Gaussian blur")
    parser.add_argument(
        "--brightness", action="store_true", help="Apply brightness scaling"
    )
    parser.add_argument("--contrast", action="store_true", help="Apply contrast change")
    parser.add_argument(
        "--jpeg", action="store_true", help="Apply JPEG compression artifacts"
    )
    args = parser.parse_args()
    main(args)
