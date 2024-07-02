import cv2
import mediapipe as mp
import numpy as np
import pickle

CONFIDENCE_THRESHOLD = 0.8

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Load the trained model
with open('models/svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to extract features
def extract_features(hand_landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

# Initialize webcam
cap = cv2.VideoCapture(0)

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
            
            # Extract features and predict gesture
            features = extract_features(hand_landmarks)
            probabilities = model.predict_proba([features])[0]
            max_prob = np.max(probabilities)
            gesture = model.classes_[np.argmax(probabilities)]
            
            # Only display the gesture if the confidence is above the threshold
            if max_prob > CONFIDENCE_THRESHOLD:
                cv2.putText(frame, f'Gesture: {gesture} ({max_prob:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Gesture: Uncertain', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display the frame
    cv2.imshow('Gesture Recognition', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()