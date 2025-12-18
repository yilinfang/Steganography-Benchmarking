import argparse
import random
import os
import subprocess
from tqdm import tqdm
import csv

seed = 42
random.seed(seed)

MAX_STRING_LENGTH = 7
SECRETS_CSV = "secrets"


def encode_image(model_path, input_image_path, save_dir, secrets, height, width):
    # Run pixi encoding process
    command = [
        "pixi",
        "run",
        "python",
        "-m",
        "stegastamp.encode_image",
        "--model",
        model_path,
        "--image",
        input_image_path,
        "--save_dir",
        save_dir,
        "--secret",
        secrets,
        "--height",
        str(height),
        "--width",
        str(width),
    ]
    # print(command)
    # print(" ".join(command))
    # result = subprocess.run(command, capture_output=True, text=True)
    subprocess.run(command)
    # print(result.stdout)


def generate_random_secret(length):
    characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(random.choice(characters) for _ in range(length))


def main(args):
    for size in tqdm(range(1, MAX_STRING_LENGTH + 1)):
        with open(f"{SECRETS_CSV}_size{size}.csv", "w") as csv_file:
            csv_file.write("secret,hidden_path\n")
            # Traverse all images in the input directory
            save_dir = os.path.join(args.output_dir, f"size_{size}")
            os.makedirs(save_dir, exist_ok=True)
            for image_name in tqdm(os.listdir(args.input_images_dir)):
                secret = generate_random_secret(size)
                input_image_path = os.path.join(args.input_images_dir, image_name)
                encode_image(
                    model_path=args.model,
                    input_image_path=input_image_path,
                    save_dir=save_dir,
                    secrets=secret,
                    height=args.height,
                    width=args.width,
                )
                csv_file.write(f"{secret},{image_name}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the model file"
    )
    parser.add_argument(
        "--input_images_dir",
        type=str,
        required=True,
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save outputs"
    )
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--secret_size", type=int, default=100)
    args = parser.parse_args()
    main(args)
