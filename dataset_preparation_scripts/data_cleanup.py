# script to clean data and remove corrupt files
import cv2
from pathlib import Path


def clean_and_standardize_dataset(dataset_path : str):
    """
    Cleans and standardizes the dataset by removing corrupt files and converting PNGs to JPGs.
    :param dataset_path:
    :return:
    """
    dataset_dir = Path(dataset_path)
    removed_count = 0
    converted_count = 0

    # iterate through all files in the dataset dir
    for file_path in dataset_dir.rglob('*'):
        if file_path.is_file():
            # read the image with unchanged channels to detect alpha layers
            img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)

            # delete the file if opencv cannot decode it
            if img is None:
                print(f"Deleting corrupt file: {file_path}")
                file_path.unlink()
                removed_count += 1
                continue

            # strip alpha channels and enforce a consistent RGB jpg format
            if len(img.shape) == 3 and img.shape[2] == 4:
                # convert 4-channel BGRA to 3-channel BGR
                img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                # save as a standard JPEG
                new_path = file_path.with_suffix('.jpg')
                cv2.imwrite(str(new_path), img_bgr)

                # delete the original PNG if a new JPEG was created
                if file_path.suffix.lower() != '.jpg':
                    file_path.unlink()
                converted_count += 1

    print(
        f"Cleanup complete. Removed {removed_count} corrupt files. Converted {converted_count} images to standard 3-channel JPEGs.")

# Note to user: This must be run before setting up PyTorch.
clean_and_standardize_dataset('../data')

import os
from pathlib import Path
from PIL import Image
import imagehash


def deduplicate_dataset(dataset_dir):
    dataset_path = Path(dataset_dir)
    seen_hashes = set()
    duplicates_removed = 0

    print("Scanning for duplicates...")
    # iterate through all images in all 4 class subdirectories
    for img_path in dataset_path.rglob('*.*'):
        if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                # open image and calculate its perceptual hash
                img = Image.open(img_path)
                img_hash = imagehash.phash(img)

                # if hash exists, it's a duplicate. Delete it.
                if img_hash in seen_hashes:
                    img_path.unlink()
                    duplicates_removed += 1
                else:
                    seen_hashes.add(img_hash)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    print(f"Deduplication complete. Removed {duplicates_removed} duplicate images.")


deduplicate_dataset('../data')