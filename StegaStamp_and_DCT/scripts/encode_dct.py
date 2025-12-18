import random
import os
import subprocess
from tqdm import tqdm
import cv2
import argparse

seed = 42
random.seed(seed)

ENCODE_SCRIPT = "../dct/run_stego_algorithm.py"
SECRETS_CSV = "secrets_dct"
INPUT_DIR = "../test_2"
# OUTPUT_DIR = "../output_3"
OUTPUT_DIR = "../output_7"
WIDTH = 224
HEIGHT = 224


def generate_random_secret(length: int) -> str:
    characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(random.choice(characters) for _ in range(length))


def encode_image(
    cover_image: str, stego_image: str, message: str, width: int, height: int
) -> None:
    command = [
        "python3",
        ENCODE_SCRIPT,
        "--cover_image",
        cover_image,
        "--stego_image",
        stego_image,
        "--message",
        message,
        "--width",
        str(width),
        "--height",
        str(height),
    ]
    # IMPORTANT: when passing a list command, use shell=False
    subprocess.run(command, shell=False, check=True)


def is_image_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def main(args):
    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    WIDTH = args.width
    HEIGHT = args.height
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = [f for f in os.listdir(INPUT_DIR) if is_image_file(f)]
    images.sort()

    for size in tqdm([1, 2, 3, 4, 5, 6, 7], desc="Secret sizes"):
        with open(f"{SECRETS_CSV}_size{size}.csv", "w") as secrets_file:
            secrets_file.write("secret,hidden_path\n")
            save_dir = os.path.join(OUTPUT_DIR, f"size_{size}")
            os.makedirs(save_dir, exist_ok=True)
            for image in tqdm(images, desc=f"Images (size={size})", leave=False):
                image_name = os.path.splitext(image)[0]
                jpeg_image_path = os.path.join(INPUT_DIR, image)
                # Read cover and resize to fixed size
                img = cv2.imread(jpeg_image_path, cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f"Failed to read image: {jpeg_image_path}")
                img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                # Save as *_input.png in the size directory
                cover_image_path = os.path.join(save_dir, f"{image_name}_input.png")
                cv2.imwrite(cover_image_path, img)
                # Generate secret and encode (encode from *_input.png for consistent residual/PSNR)
                secret = generate_random_secret(size)
                stego_image_path = os.path.join(save_dir, f"{image_name}_hidden.png")
                encode_image(cover_image_path, stego_image_path, secret, WIDTH, HEIGHT)
                # Residual: abs difference between input and hidden
                hidden = cv2.imread(stego_image_path, cv2.IMREAD_COLOR)
                if hidden is None:
                    raise RuntimeError(
                        f"Failed to read stego image: {stego_image_path}"
                    )
                if hidden.shape != img.shape:
                    # Safety: force same size if the stego script outputs a different size
                    hidden = cv2.resize(
                        hidden, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA
                    )
                residual = cv2.absdiff(img, hidden)
                residual_png_path = os.path.join(save_dir, f"{image_name}_residual.png")
                cv2.imwrite(residual_png_path, residual)
                # Record secret and path
                secrets_file.write(f"{secret},{image_name}_hidden.png\n")


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--input_dir", type=str, default=INPUT_DIR)
    argparse.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    argparse.add_argument("--width", type=int, default=WIDTH)
    argparse.add_argument("--height", type=int, default=HEIGHT)
    args = argparse.parse_args()
    main(args)
