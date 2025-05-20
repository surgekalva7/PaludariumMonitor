import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model

# Load the pre-trained MobileNetV2 model (replace with your fine-tuned model if available)
model = MobileNetV2(weights='imagenet')

# Define a function to preprocess the frame
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224))  # Resize to 224x224
    array_frame = np.expand_dims(resized_frame, axis=0)  # Add batch dimension
    preprocessed_frame = preprocess_input(array_frame)  # Preprocess for MobileNetV2
    return preprocessed_frame

# Start video capture
cap = cv2.VideoCapture(0)  # 0 for default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)

    # Predict the class
    predictions = model.predict(preprocessed_frame)
    decoded_predictions = decode_predictions(predictions, top=1)[0]  # Get top prediction
    label = decoded_predictions[0][1]  # Get class label
    confidence = decoded_predictions[0][2]  # Get confidence score

    # Overlay the prediction on the frame
    text = f"{label}: {confidence:.2f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Plant Identifier", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()