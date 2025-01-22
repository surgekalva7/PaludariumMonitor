from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.yaml")

results = model.train(data = "data.yaml", epochs=50, imgsz=640, device='0,1,2,3')

print(model.info())

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read camera")
        exit()

    # pass frame through model
    frame_resized = cv2.resize(frame, (640, 480))
    res = model.predict(source=frame_resized, show=True, conf=0.45)

    #Display resulting frame
    cv2.imshow('Stream', res[0].plot())

    # Break loop on 'q' for quit
    if cv2.waitKey(1) == ord('q'):
        break