import os
import random
import shutil

# -------- configuration --------
SRC_DIR = "train2017"
TEST_SRC_DIR = "test"
TRAIN_DIR = "train"
VAL_DIR = "val"
TEST_DIR = "test_2"
TRAIN_RATIO = 0.8
TEST_SIZE = 100
RANDOM_SEED = 42
# -------------------------------

random.seed(RANDOM_SEED)

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# list image files
# images = [
#     f for f in os.listdir(SRC_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))
# ]
#
# images.sort()
# random.shuffle(images)
#
# num_train = int(len(images) * TRAIN_RATIO)
# train_images = images[:num_train]
# val_images = images[num_train:]
#
# list test image files
test_images = [
    f for f in os.listdir(TEST_SRC_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
test_images.sort()
random.shuffle(test_images)
test_images = test_images[:TEST_SIZE]


def move_files(file_list, src_dir, dst_dir):
    for fname in file_list:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        shutil.move(src, dst)


def copy_files(file_list, src_dir, dst_dir):
    for fname in file_list:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        shutil.copy2(src, dst)


# move_files(train_images, SRC_DIR, TRAIN_DIR)
# move_files(val_images, SRC_DIR, VAL_DIR)
move_files(test_images, TEST_SRC_DIR, TEST_DIR)
# print(f"Total images: {len(images)}")
# print(f"Train images: {len(train_images)}")
# print(f"Val images:   {len(val_images)}")
print(f"Test images:  {len(test_images)}")
