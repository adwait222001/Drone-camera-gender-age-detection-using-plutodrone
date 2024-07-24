import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from plutocontrol import pluto

drone = pluto()

drone.cam()
drone.connect()
# Initialize HandDetector from cvzone with higher confidence detection
detector = HandDetector(detectionCon=0.9, maxHands=2)  # Detect up to 2 hands

# Function to determine if the palm or back side of the hand is facing the camera
def get_hand_orientation(lmList, handType):
    if len(lmList) == 21:
        # Get key landmarks for palm orientation
        wrist = np.array(lmList[0])
        thumb_tip = np.array(lmList[4])
        index_tip = np.array(lmList[8])
        middle_tip = np.array(lmList[12])
        
        # Calculate vectors from wrist to finger tips
        vector_thumb = thumb_tip - wrist
        vector_index = index_tip - wrist
        vector_middle = middle_tip - wrist
        
        # Calculate the normal of the palm plane using cross product
        palm_normal = np.cross(vector_thumb, vector_index)
        palm_normal = palm_normal / np.linalg.norm(palm_normal)  # Normalize
        
        # Assuming camera axis is along Z direction
        camera_axis = np.array([0, 0, 1])
        
        # Calculate dot product to find the alignment with the camera axis
        dot_product = np.dot(palm_normal, camera_axis)
        
        # For the right hand, the detection is inverted
        if handType == "Right":
            dot_product = -dot_product
        
        # Palm facing the camera if dot product is close to 1 (parallel to Z-axis)
        if dot_product > 0.7:  # Tolerance for direction alignment
            return "Palm Facing Camera"
        else:
            return "Back Facing Camera"
    return "Unknown"

# Open webcam
cap = cv2.VideoCapture(0)

# Variables to store the last printed command
last_command = None

while True:
    # Read frame from webcam
    success, img = cap.read()
    if not success:
        break

    # Detect hands in the frame
    hands, img = detector.findHands(img)

    num_hands = len(hands)  # Number of detected hands

    if num_hands > 0:
        orientations = []
        for hand in hands:
            lmList = hand['lmList']  # List of landmarks for the detected hand
            handType = hand['type']  # 'Right' or 'Left'
            
            # Determine hand orientation
            orientation = get_hand_orientation(lmList, handType)
            orientations.append(orientation)
        
        # Check if both hands are detected and their orientations
        if num_hands == 2:
            if all(o == "Palm Facing Camera" for o in orientations):
                current_command = "arm"
            elif all(o == "Back Facing Camera" for o in orientations):
                current_command = "disarm"
            else:
                current_command = None
        else:
            current_command = None

        # Check for specific condition to print "take off"
        if num_hands == 1:
            right_hand = next((hand for hand in hands if hand['type'] == 'Right'), None)
            left_hand = next((hand for hand in hands if hand['type'] == 'Left'), None)
            if right_hand:
                orientation = get_hand_orientation(right_hand['lmList'], 'Right')
                if orientation == "Back Facing Camera":
                    current_command = "take off"
                else:
                    current_command = None
            elif left_hand:
                orientation = get_hand_orientation(left_hand['lmList'], 'Left')
                if orientation == "Back Facing Camera":
                    current_command = "land"
                else:
                    current_command = None
            else:
                current_command = None

        # Print the command only if it's different from the last printed command
        if current_command and current_command != last_command:
            print(current_command)
            last_command = current_command

        # Display messages on the frame
        for i, hand in enumerate(hands):
            message = f"{hand['type'].capitalize()} Hand: {orientations[i]}"
            cv2.putText(img, message, (50, 50 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(img, "Please show at least two hands", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Hand Gesture Recognition", img)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
