import os
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Path to the dataset
data_dir = "data/train"  # Replace with the path to your training dataset
class_names_file = "class_names.txt"  # File to save class names

# Load the dataset
dataset = image_dataset_from_directory(
    data_dir,
    image_size=(224, 224),
    batch_size=32
)

# Get class names
class_names = dataset.class_names

# Save class names to a file
with open(class_names_file, "w") as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")

print(f"Class names saved to '{class_names_file}'")