import random
import os
import subprocess
from tqdm import tqdm
import cv2
import argparse
import csv


DECODE_SCRIPT = "../dct/extract_stego_image.py"
SECRETS_CSV = "secrets_dct_size7.csv"
WIDTH = 224
HEIGHT = 224

secrets_reader = csv.reader(open(SECRETS_CSV, "r"))
# Skip the header
next(secrets_reader)
secrets_map = {row[1]: row[0] for row in secrets_reader}


def decode_image(input_image_path: str, ground_truth: str) -> float:
    command = [
        "python3",
        DECODE_SCRIPT,
        "--stego_image",
        input_image_path,
        "--ground_truth",
        ground_truth,
    ]
    print(" ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    # If the command fails, return 0.0
    if result.returncode != 0:
        return 0.0
    # If `Recovery rate:` is in the output, get the recovery rate
    if "Recovery rate:" in result.stdout:
        recovery_rate = result.stdout.split("Recovery rate:")[1].split()[0]
        # Remove the trailing `%` from the recovery rate
        recovery_rate = float(recovery_rate.replace("%", "")) / 100.0
        print(f"Recovery rate for {input_image_path}: {recovery_rate}")
        return recovery_rate
    else:
        return 1.0  # Successfully decoded, so recovery rate is 1.0


def is_image_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def get_secret(input_file: str) -> str:
    input_file_name = str(os.path.splitext(input_file)[0])
    print(f"Getting secret for {input_file_name}")
    # PREFIX = "../output_3/size_7/"
    key = input_file_name + ".png"
    print(f"Secret for {key}: {secrets_map.get(key, '')}")
    return secrets_map.get(key, "")


def main(args):
    input_dir = args.input_dir
    # Only process the images end with _hidden.png
    input_files = [
        f
        for f in os.listdir(input_dir)
        if is_image_file(f) and f.endswith("_hidden.png")
    ]
    total_recovery_rate = 0.0
    for input_file in tqdm(input_files):
        secrets = get_secret(input_file)
        print(f"Decoding {input_file} with secret {secrets}")
        input_image_path = os.path.join(input_dir, input_file)
        recovery_rate = decode_image(input_image_path, secrets)
        total_recovery_rate += recovery_rate
    average_recovery_rate = total_recovery_rate / len(input_files)
    print(f"Average recovery rate: {average_recovery_rate:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory")
    args = parser.parse_args()
    main(args)
