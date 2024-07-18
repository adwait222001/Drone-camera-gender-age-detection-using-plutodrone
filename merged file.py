import cv2
import numpy as np
import keyboard as kb
from plutocontrol import pluto
from multiprocessing import Process, Queue
import threading

# Initialize Pluto and keyboard mappings
my_pluto = pluto()
keyboard_cmds = {
    '[A': 10, '[D': 30, '[C': 40, 'w': 50, 's': 60, ' ': 70, 'r': 80, 't': 90,
    'p': 100, '[B': 110, 'n': 120, 'q': 130, 'e': 140, 'a': 150, 'd': 160,
    '+': 15, '1': 25, '2': 30, '3': 35, '4': 45, 'c': 200, 'x': 210
}

key_map = {'up': '[A', 'down': '[B', 'left': '[D', 'right': '[C', 'space': ' '}

# Model files and mean values for gender detection
FACE_PROTO = "opencv_face_detector.pbtxt"
FACE_MODEL = "opencv_face_detector_uint8.pb"
GENDER_PROTO = "gender_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"

face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
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

# Function to process and display gender detection on a frame
def process_frame(frame):
    face_boxes = detect_faces(frame)
    for (startX, startY, endX, endY) in face_boxes:
        face = frame[startY:endY, startX:endX]
        if face.size == 0:
            continue

        gender = predict_gender(face)
        label = f"{gender}"
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame

# Function to read keyboard input
def getKey(queue):
    while True:
        event = kb.read_event()
        key = key_map.get(event.name, event.name) if event.event_type == kb.KEY_DOWN else None
        if key:
            queue.put(key)

# Function to control Pluto based on keyboard input
def control_pluto(queue):
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

    while True:
        key = queue.get()
        actions.get(keyboard_cmds.get(key, 80), my_pluto.reset)()

if __name__ == '__main__':
    queue = Queue()

    # Create thread for keyboard input and process for Pluto control
    keyboard_thread = threading.Thread(target=getKey, args=(queue,))
    pluto_process = Process(target=control_pluto, args=(queue,))

    # Start thread for keyboard input
    keyboard_thread.start()

    # Start process for Pluto control
    pluto_process.start()

    # Start webcam feed for gender detection
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame for gender detection
        processed_frame = process_frame(frame.copy())
        cv2.imshow('Gender Detection', processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Join keyboard thread and terminate Pluto process
    keyboard_thread.join()
    pluto_process.terminate()
