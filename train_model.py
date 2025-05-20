import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Paths
data_dir = "data"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "validation")
model_path = "plant_classifier_model.h5"

def train_model():
    """
    Trains a plant classification model using the dataset in the 'data/train' and 'data/validation' directories.
    Saves the trained model to 'plant_classifier_model.h5'.
    """
    # Load datasets
    train_dataset = image_dataset_from_directory(
        train_dir,
        image_size=(224, 224),
        batch_size=32
    )
    val_dataset = image_dataset_from_directory(
        val_dir,
        image_size=(224, 224),
        batch_size=32
    )

    # Extract class names before applying transformations
    class_names = train_dataset.class_names

    # Normalize pixel values to [0, 1]
    train_dataset = train_dataset.map(lambda x, y: (x / 255.0, y))
    val_dataset = val_dataset.map(lambda x, y: (x / 255.0, y))

    # Load the pre-trained MobileNetV2 model
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    # Freeze the base model
    base_model.trainable = False

    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    predictions = Dense(len(class_names), activation="softmax")(x)

    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10
    )

    # Save the trained model
    model.save(model_path)
    print(f"Model training complete and saved as '{model_path}'")