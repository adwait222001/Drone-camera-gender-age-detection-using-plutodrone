import threading
import keyboard as kb
from plutocontrol import pluto
import cv2
import numpy as np

# Global variable for pluto object
my_pluto = None

def control_pluto():
    global my_pluto
    my_pluto = pluto()
    my_pluto.cam()
    actions = {
        70: lambda: my_pluto.disarm() if my_pluto.rcAUX4 == 1500 else my_pluto.arm(),
        10: my_pluto.forward,
        30: my_pluto.left,
        40: my_pluto.right,
        80: my_pluto.reset,
        50: my_pluto.increase_height,
        60: my_pluto.decrease_height,
        110: my_pluto.backward,
        130: my_pluto.take_off,
        140: my_pluto.land,
        150: my_pluto.left_yaw,
        160: my_pluto.right_yaw,
        120: lambda: (print("Developer Mode ON"), setattr(my_pluto, 'rcAUX2', 1500)),
        200: my_pluto.connect,
        210: my_pluto.disconnect,
    }

    keyboard_cmds = {
        '[A': 10, '[D': 30, '[C': 40, 'w': 50, 's': 60, ' ': 70, 'r': 80, 't': 90,
        'p': 100, '[B': 110, 'n': 120, 'q': 130, 'e': 140, 'a': 150, 'd': 160,
        '+': 15, '1': 25, '2': 30, '3': 35, '4': 45, 'c': 200, 'x': 210
    }

    key_map = {'up': '[A', 'down': '[B', 'left': '[D', 'right': '[C', 'space': ' '}

    def getKey():
        event = kb.read_event()
        return key_map.get(event.name, event.name) if event.event_type == kb.KEY_DOWN else None

    while (key := getKey()) != 'e':
        actions.get(keyboard_cmds.get(key, 80), my_pluto.reset)()
    print("stopping")

def gender_detection():
    global my_pluto

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
            print(label)  # Print identified gender
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Disarm if gender is male, arm if gender is female
            if gender == 'Male':
                my_pluto.disarm()
            else:
                my_pluto.arm()

        # Display the live feed with gender detection
        cv2.imshow('Gender Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create threads
    pluto_thread = threading.Thread(target=control_pluto)
    gender_thread = threading.Thread(target=gender_detection)

    # Start threads
    pluto_thread.start()
    gender_thread.start()

    # Join threads
    pluto_thread.join()
    gender_thread.join()
