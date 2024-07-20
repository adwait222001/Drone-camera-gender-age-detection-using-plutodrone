import cv2
import numpy as np

# Model files and mean values
FACE_PROTO = "opencv_face_detector.pbtxt"
FACE_MODEL = "opencv_face_detector_uint8.pb"
GENDER_PROTO = "gender_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"

# Load models
face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

# Mean values for normalization
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDERS = ['Male', 'Female']

# Function to detect faces in a frame
def detect_faces(frame, confidence_threshold=0.7):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), MODEL_MEAN_VALUES, swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()
    face_boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            face_boxes.append(box.astype(int))

    return face_boxes

# Function to predict gender of a face
def predict_gender(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    return GENDERS[gender_preds[0].argmax()]

# Initialize webcam
cap = cv2.VideoCapture("http://127.0.0.1:5000/video_feed")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame for gender detection
    face_boxes = detect_faces(frame)
    for (startX, startY, endX, endY) in face_boxes:
        face = frame[startY:endY, startX:endX]
        if face.size == 0:
            continue
        
        # Ensure the face region is properly resized
        face = cv2.resize(face, (227, 227))
        
        gender = predict_gender(face)
        label = f"{gender}"
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the live feed with gender detection
    cv2.imshow('Gender Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
