import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split


def create_frozen_splits(source_dir, output_dir, test_size=0.15, val_size=0.15):
    source_path = Path(source_dir)
    classes = ['dry', 'wet', 'recyclable', 'ewaste']

    # gather all file paths and their corresponding labels
    all_files = []
    all_labels = []

    for cls in classes:
        cls_dir = source_path / cls
        if not cls_dir.exists():
            continue
        for img_path in cls_dir.glob('*.*'):
            if img_path.is_file():
                all_files.append(img_path)
                all_labels.append(cls)

    # first split: Separate out the test set (15%)
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        all_files, all_labels,
        test_size=test_size,
        stratify=all_labels,
        random_state=42
    )

    # second Split: Separate Train and Validation sets
    # adjust validation ratio based on the remaining training data size
    val_ratio = val_size / (1.0 - test_size)
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_val_files, train_val_labels,
        test_size=val_ratio,
        stratify=train_val_labels,
        random_state=42
    )

    # helper function to copy files to their new split directories
    def copy_files(file_list, label_list, split_name):
        print(f"Copying {len(file_list)} files to {split_name} set...")
        for src_file, label in zip(file_list, label_list):
            dest_dir = Path(output_dir) / split_name / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dest_dir / src_file.name)

    # execute the copies
    copy_files(train_files, train_labels, 'train')
    copy_files(val_files, val_labels, 'val')
    copy_files(test_files, test_labels, 'test')

    print("Physical splitting complete. The test set is now frozen.")

create_frozen_splits('./data', 'data')

# this overwrites the older data folder.