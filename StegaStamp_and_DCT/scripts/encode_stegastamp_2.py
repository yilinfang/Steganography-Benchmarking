import argparse
import os
import random
import string
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

SEED = 42
MAX_STRING_LENGTH = 7
CHARS = string.ascii_letters + string.digits


def encode_image(model_path, input_image_path, save_dir, secret, height, width):
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
        secret,
        "--height",
        str(height),
        "--width",
        str(width),
    ]
    # Use check=True so failures raise and we can record them
    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def deterministic_secret(size, image_name, seed=SEED):
    # Make the secret stable regardless of parallel execution order
    rng = random.Random(f"{seed}:{size}:{image_name}")
    return "".join(rng.choice(CHARS) for _ in range(size))


def worker(job):
    # Defined at top-level so it is picklable for ProcessPoolExecutor
    (size, image_name, model, input_dir, output_dir, height, width) = job
    save_dir = os.path.join(output_dir, f"size_{size}")
    os.makedirs(save_dir, exist_ok=True)

    secret = deterministic_secret(size, image_name)
    input_image_path = os.path.join(input_dir, image_name)
    encode_image(model, input_image_path, save_dir, secret, height, width)
    return (size, image_name)


def main(args):
    image_names = [
        n
        for n in os.listdir(args.input_images_dir)
        if os.path.isfile(os.path.join(args.input_images_dir, n))
    ]

    jobs = []
    for size in range(1, MAX_STRING_LENGTH + 1):
        for image_name in image_names:
            jobs.append(
                (
                    size,
                    image_name,
                    args.model,
                    args.input_images_dir,
                    args.output_dir,
                    args.height,
                    args.width,
                )
            )

    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker, j) for j in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Encoding"):
            try:
                fut.result()
            except subprocess.CalledProcessError as e:
                failures.append(e.stderr)
            except Exception as e:
                failures.append(repr(e))

    if failures:
        print(f"\nDone with {len(failures)} failures. First failure:\n{failures[0]}")
    else:
        print("\nDone with 0 failures.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input_images_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2)
    )
    args = parser.parse_args()
    main(args)
