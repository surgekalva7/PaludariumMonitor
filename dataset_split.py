import os
import shutil
import random

# Paths to the dataset
data_dir = "data"
no_aug_dir = os.path.join(data_dir, "Plant_leave_diseases_dataset_with_augmentation")
aug_dir = os.path.join(data_dir, "Plant_leave_diseases_dataset_without_augmentation")
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "validation")

# Create train and validation directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Split ratio
train_ratio = 0.8

# Function to merge and split datasets
def merge_and_split_datasets(no_aug_dir, aug_dir, train_dir, val_dir, train_ratio):
    # Get class folders from one of the directories (assuming both have the same structure)
    classes = [d for d in os.listdir(no_aug_dir) if os.path.isdir(os.path.join(no_aug_dir, d))]

    for class_name in classes:
        # Paths to class folders
        no_aug_class_dir = os.path.join(no_aug_dir, class_name)
        aug_class_dir = os.path.join(aug_dir, class_name)

        # Create class subfolders in train and validation directories
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        # Collect all images from both folders
        images = []
        if os.path.exists(no_aug_class_dir):
            images += [os.path.join(no_aug_class_dir, f) for f in os.listdir(no_aug_class_dir) if os.path.isfile(os.path.join(no_aug_class_dir, f))]
        if os.path.exists(aug_class_dir):
            images += [os.path.join(aug_class_dir, f) for f in os.listdir(aug_class_dir) if os.path.isfile(os.path.join(aug_class_dir, f))]

        # Shuffle the images
        random.shuffle(images)

        # Split the images into train and validation sets
        train_count = int(len(images) * train_ratio)
        train_images = images[:train_count]
        val_images = images[train_count:]

        # Move images to train folder
        for img_path in train_images:
            dest = os.path.join(train_dir, class_name, os.path.basename(img_path))
            shutil.copy(img_path, dest)

        # Move images to validation folder
        for img_path in val_images:
            dest = os.path.join(val_dir, class_name, os.path.basename(img_path))
            shutil.copy(img_path, dest)

# Merge and split the datasets
merge_and_split_datasets(no_aug_dir, aug_dir, train_dir, val_dir, train_ratio)

print("Dataset has been merged, split, and organized into train and validation sets!")