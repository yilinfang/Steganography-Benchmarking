import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import csv

SECRETS_CSV = "secrets_size7.csv"
secrets_reader = csv.reader(open(SECRETS_CSV, "r"))
# Skip the header
next(secrets_reader)
secrets_map = {row[1]: row[0] for row in secrets_reader}


def decode_image(
    model_path, input_image_path, height, width, secret_size, ground_truth=None
):
    # Run pixi decoding process
    command = [
        "pixi",
        "run",
        "python",
        "-m",
        "stegastamp.decode_image",
        "--model",
        model_path,
        "--image",
        input_image_path,
        "--height",
        str(height),
        "--width",
        str(width),
        "--secret_size",
        str(secret_size),
    ]
    # print(command)
    # print(" ".join(command))
    # result = subprocess.run(command, capture_output=True, text=True)
    # print(result.stdout)
    result = subprocess.run(command, capture_output=True, text=True)
    # If `decode failed` is shown in the output, return false
    if "decode failed" in result.stdout:
        return 0.0
    # return True
    if ground_truth is not None:
        if ground_truth == result.stdout:
            return 1.0
        else:
            # Compare the decoded secret with the ground truth
            recovery_rate = sum(
                1 for x, y in zip(ground_truth, result.stdout) if x == y
            ) / len(ground_truth)
            print(f"Ground truth: {ground_truth}")
            print(f"Decoded: {result.stdout}")
            print(f"Recovery rate: {recovery_rate:.4f}")
            return recovery_rate
    return 0.0


def get_secret(input_file: str) -> str:
    input_file_name = str(os.path.splitext(input_file)[0])
    # Keep only the file name without extension
    input_file_name = os.path.basename(input_file_name)
    # Remove _hidden
    input_file_name = input_file_name.replace("_hidden", "")
    # print(f"Getting secret for {input_file_name}")
    # PREFIX = "../output_4/size_7/"
    # PREFIX = "workspace/coco/test_2/"
    # PREFIX = ""
    key = input_file_name + ".jpg"
    # print(f"Secret for {key}: {secrets_map.get(key, '')}")
    return secrets_map.get(key, "")


def worker(job):
    model, image_path, height, width, secret_size = job
    ground_truth = get_secret(image_path)
    ok = decode_image(model, image_path, height, width, secret_size, ground_truth)
    return ok


def main(args):
    input_images = [
        os.path.join(args.input_images_dir, f)
        for f in os.listdir(args.input_images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
        and os.path.isfile(os.path.join(args.input_images_dir, f))
    ]

    total_count = len(input_images)
    success_count = 0

    jobs = [
        (args.model, p, args.height, args.width, args.secret_size) for p in input_images
    ]

    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker, j) for j in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Decoding"):
            try:
                if fut.result():
                    success_count += fut.result()
            except Exception:
                failures += 1

    print(
        f"Decoding success rate: {success_count}/{total_count} = {success_count / total_count:.4%}"
    )
    if failures:
        print(f"Note: {failures} jobs raised exceptions (counted as failure).")


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
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--secret_size", type=int, default=100)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2)
    )
    args = parser.parse_args()
    main(args)
