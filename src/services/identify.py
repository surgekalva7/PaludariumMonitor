# CV captures live feed
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

# Load pre-trained MobileNetV2 model
def load_model():
    print("Loading MobileNetV2 model...")
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    return model

# Preprocess each frame for the model - Cleans the raw data info usuable data
def preprocess_frame(frame):
    # Resize frame to 224x224 (MobileNetV2 input size)
    img = cv2.resize(frame, (224, 224))
    img_array = np.expand_dims(img, axis=0)  # Add batch dimension
    img_array = mobilenet_v2.preprocess_input(img_array)  # Preprocess for MobileNetV2
    return img_array

# Predict and decode results for each frame
def predict_frame(model, frame):
    preprocessed = preprocess_frame(frame)
    predictions = model.predict(preprocessed)
    decoded_predictions = decode_predictions(predictions, top=3)[0]  # Top 3 predictions
    return decoded_predictions

# Annotate predictions on the frame
def annotate_frame(frame, predictions):
    y_offset = 20
    for i, (imagenet_id, label, confidence) in enumerate(predictions):
        text = f"{label}: {confidence * 100:.2f}%"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 20
    return frame

def main():
    # Load the pre-trained model
    model = load_model()

    # Open video stream (camera feed)
    # There can be multiple places to capture video feed
    cap = cv2.VideoCapture(0)  # 0 for the default webcam

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Predict the current frame
        predictions = predict_frame(model, frame)

        # Annotate the frame with predictions
        frame = annotate_frame(frame, predictions)

        # Display the annotated frame
        cv2.imshow("Live Plant/Organism/Bacteria Identifier", frame)

        # Exit on 'q' key press
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    # Release the video stream and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()