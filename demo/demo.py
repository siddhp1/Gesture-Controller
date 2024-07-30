import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the trained model and label encoder
model = load_model("../models/HaGRID/second/model.keras")
le_classes = np.load(
    "../models/HaGRID/second/label_encoder_classes.npy", allow_pickle=True
)
le = LabelEncoder()
le.classes_ = le_classes

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Start webcam feed
cap = cv2.VideoCapture(0)

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
            landmarks = [
                coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)
            ]
            landmarks = np.array(landmarks).reshape(1, 21, 3)

            # Make prediction
            prediction = model.predict(landmarks)
            predicted_label_index = np.argmax(prediction)
            gesture = le.inverse_transform([predicted_label_index])[0]
            confidence = prediction[0][predicted_label_index]

            # Draw landmarks and display the detected gesture with confidence
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(
                frame,
                f"{gesture} ({confidence:.2f})",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    # Display the frame
    cv2.imshow("Gesture Detection", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
