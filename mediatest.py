import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pyautogui
import time

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype()")

# Load the trained model and label encoder
model = load_model('models/HaGRID/second/model.keras')
le_classes = np.load('models/HaGRID/second/label_encoder_classes.npy', allow_pickle=True)
le = LabelEncoder()
le.classes_ = le_classes

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Start webcam feed
cap = cv2.VideoCapture(0)
print("STARTED")

gesture_action_map = {
    "palm": "playpause",
    "fist": "playpause",
    "like": "volumeup",
    "dislike": "volumedown",
    "one": "volumemute"
}

gesture_hold_time = 1.0  # Time in seconds to hold the gesture
gesture_start_time = None
current_gesture = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect hands
    result = hands.process(frame_rgb)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract landmarks
            landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
            landmarks = np.array(landmarks).reshape(1, 21, 3)

            # Make prediction
            prediction = model.predict(landmarks)
            predicted_label_index = np.argmax(prediction)
            gesture = le.inverse_transform([predicted_label_index])[0]
            confidence = prediction[0][predicted_label_index]
            
            # Output information to console
            print(f'{gesture} ({confidence:.2f})')
            
            if confidence > 0.8:
                if current_gesture == gesture:
                    if time.time() - gesture_start_time >= gesture_hold_time:
                        action = gesture_action_map.get(gesture)
                        if action:
                            pyautogui.press(action)
                            print(f'Performed action: {action}')
                            gesture_start_time = None  # Reset to prevent repeated actions
                            current_gesture = None
                else:
                    current_gesture = gesture
                    gesture_start_time = time.time()
            else:
                current_gesture = None
                gesture_start_time = None
    else:
        current_gesture = None
        gesture_start_time = None

    # Display the resulting frame
    #cv2.imshow('Gesture Control', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
