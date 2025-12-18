# Benchmarking StegaStamp and DCT

## Setup

0. Ensure that you have Pixi installed.
1. Unzip the modified StegaStamp code in `./assets/stegastamp.zip` to `./stegastamp/`. The original code is available at this [link](https://github.com/jsrdcht/StegaStamp-pytorch).
2. Unzip the modified DCT code in `./assets/dct.zip` to `./dct/`. The original code is available at this [link](https://github.com/MasonEdgar/DCT-Image-Steganography).

## Training StegaStamp

Before training, please download the COCO 2017 dataset, and put the training images in `./workspace/coco/train` and validation images in `./workspace/coco/val`.

```bash
# All the scripts need to be run from the `./stegastamp/` directory
cd stegastamp/
# Activate the pixi venv
pixi shell
# Execute the training script
bash ../scripts/train_coco.sh
# The trained model will be saved in `./stegastamp/checkpoints` and the training logs will be saved int `./stegastamp/output`
```

## Benchmarking StegaStamp

### Example 1: PSNR benchmark

```bash
# All the scripts need to be run from the `./stegastamp/` directory
cd stegastamp/
# Activate the pixi venv
pixi shell
# Encode the images with random fix-sized payloads
python3 ../scripts/encode_stegastamp.py --model ../assets/best.pth --input_images_dir ../../Data/demo_images_small --output_dir ../workspace/stegastamp_output/
# Benchmark the PSNR of the encoded images of each payload size
python3 ../scripts/bench_psnr.py ../workspace/stegastamp_output/
```

### Example 2: Rotation benchmark

```bash
# All the scripts need to be run from the `./stegastamp/` directory
cd stegastamp/
# Activate the pixi venv
pixi shell
# Copy the encoded images from the previous step, we use 7bytes(56 bits) as example (skip if you already have the encoded images).
mkdir -p ../workspace/stegastamp_output/initial/
cp ../workspace/stegastamp_output/size_7/*_hidden.png ../workspace/stegastamp_output/initial/
# Add noice to the encoded images
# You could also add other type of noises like gaussian noise, gaussian blur by adding `--gaussian` or `--blur` arguments.
# After this step, the noised images will be saved in `../workspace/stegastamp_output/rotate/angle_1/`, `../workspace/stegastamp_output/rotate/angle_2/` etc...
python3 ../scripts/addnoice.py --input_dir ../workspace/stegastamp_output/initial/ --output_dir ../workspace/stegastamp_output/  --rotate
# Decode the noised images and get the accuracy
# You could also change the `angle_1` to other angles like `angle_2`, `angle_3` etc. to benchmark other rotation angles.
python3 ../scripts/decode_stegastamp_2.py --model ../assets/best.pth --workers 4 --input_images_dir ../workspace/stegastamp_output/rotate/angle_1/
```

## Benchmarking DCT

### Example 1: PSNR benchmark

```bash
# All the scripts need to be run from the `./stegastamp/` directory
cd ./stegastamp/
# Activate the pixi venv
pixi shell
# Encode the images with random fix-sized payloads
python3 ../scripts/encode_dct.py --input_dir ../../Data/demo_images_small --output_dir ../workspace/dct_output/
# Benchmark the PSNR of the encoded images of each payload size
python3 ../scripts/bench_psnr.py ../workspace/dct_output/
```

### Example 2: Rotation benchmark

```bash
# All the scripts need to be run from the `./stegastamp/` directory
cd ./stegastamp/
# Activate the pixi venv
pixi shell
# Copy the encoded images from the previous step, we use 7bytes(56 bits) as example (skip if you already have the encoded images).
mkdir -p ../workspace/dct_output/initial/
cp ../workspace/dct_output/size_7/*_hidden.png ../workspace/dct_output/initial/
# Add noice to the encoded images
# You could also add other type of noises like gaussian noise, gaussian blur by adding `--gaussian` or `--blur` arguments.
# After this step, the noised images will be saved in `../workspace/dct_output/rotate/angle_1/`, `../workspace/dct_output/rotate/angle_2/` etc...
python3 ../scripts/addnoice.py --input_dir ../workspace/dct_output/initial/ --output_dir ../workspace/dct_output/  --rotate
# Decode the noised images and get the accuracy
# You could also change the `angle_1` to other angles like `angle_2`, `angle_3` etc. to benchmark other rotation angles.
python3 ../scripts/decode_dct.py --input_dir ../workspace/dct_output/rotate/angle_1/
```
