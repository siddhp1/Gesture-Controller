import cv2
import mediapipe as mp
import pickle
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Function to save landmarks
def save_landmarks(name, landmarks_list):
    with open(f'data/raw/{name}.pkl', 'ab') as f:
        for landmarks in landmarks_list:
            pickle.dump(landmarks, f)

# Initialize webcam
cap = cv2.VideoCapture(0)

gesture_data = []
collecting = False
start_time = 0
gesture_name = "fist"  # The name of the gesture being recorded

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    result = hands.process(rgb_frame)

    # Check if hands are detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            if collecting:
                gesture_data.append(hand_landmarks)
                elapsed_time = time.time() - start_time
                cv2.putText(frame, f'Collecting data: {elapsed_time:.1f}s', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if elapsed_time >= 30:
                    collecting = False
                    save_landmarks(gesture_name, gesture_data)
                    gesture_data = []
                    cv2.putText(frame, 'Data collection completed', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display instructions
    cv2.putText(frame, 'Press "c" to start collecting data', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, 'Hold the gesture for 30 seconds', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Start data collection with 'c' key
    if cv2.waitKey(1) & 0xFF == ord('c'):
        collecting = True
        start_time = time.time()
        gesture_data = []
    
    # Display the frame
    cv2.imshow('Gesture Data Collection', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()