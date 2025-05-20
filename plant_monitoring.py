import os
import cv2
import numpy as np
import argparse
from tensorflow.keras.models import load_model
from train_model import train_model

# Paths
model_path = "plant_classifier_model.h5"
class_names_path = "class_names.txt"

def load_class_names():
    """
    Loads class names from the 'class_names.txt' file.
    """
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(f"Class names file '{class_names_path}' not found. Train the model first.")
    
    with open(class_names_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

def open_camera():
    """
    Opens the camera and uses the trained model to identify plants in real-time.
    """
    # Load the trained model
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Train the model first.")
        return

    model = load_model(model_path)

    # Load class names
    class_names = load_class_names()

    # Start video capture
    cap = cv2.VideoCapture(0)  # 0 for default camera

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            continue

        # Preprocess the frame
        resized_frame = cv2.resize(frame, (224, 224))
        array_frame = np.expand_dims(resized_frame, axis=0)
        preprocessed_frame = array_frame / 255.0  # Normalize

        # Predict the class
        predictions = model.predict(preprocessed_frame)
        class_idx = np.argmax(predictions)
        confidence = predictions[0][class_idx]
        class_name = class_names[class_idx]  # Get class name from the loaded list

        # Overlay the prediction on the frame
        text = f"{class_name}: {confidence:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Plant Identifier", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plant Monitoring System")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "camera"], help="Mode: 'train' to train the model, 'camera' to open the camera")
    args = parser.parse_args()

    if args.mode == "train":
        train_model()  # Call the training function
    elif args.mode == "camera":
        open_camera()