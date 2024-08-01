# ML Libraries
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model  # Switch to TensorFlow Lite after
from sklearn.preprocessing import LabelEncoder

# Other
import pyautogui
import time
import json
import logging
import math


class GestureController:
    def __init__(self, config_path):
        self.config_path = config_path

        # Call function to load sklearn and mp resources
        self.load_resources()

        # Call function to load the config file
        self.load_config()

        # Logging
        logging.basicConfig(filename="gesture_controller.log", level=logging.INFO)

        # Variables
        self.current_gesture = None
        self.gesture_start_time = None

    def load_resources(self):
        # Load model and labels
        self.model = load_model("../models/HaGRID/second/model.keras")
        le_classes = np.load(
            "../models/HaGRID/second/label_encoder_classes.npy", allow_pickle=True
        )
        self.le = LabelEncoder()
        self.le.fit(le_classes)

        # Load mediapipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def load_config(self):
        # Load the json into a dictionary
        with open(self.config_path, "r") as config_file:
            config = json.load(config_file)

        # Extract values, and provide defaults if not found
        self.gesture_action_map = config.get(
            "gesture_action_map",
            {
                "palm": "playpause",
                "fist": "playpause",
                "like": "volumeup",
                "dislike": "volumedown",
                "one": "volumemute",
            },
        )
        self.gesture_hold_time = config.get("gesture_hold_time", 1.0)
        self.confidence_threshold = config.get("confidence_threshold", 0.8)
        self.camera_index = config.get("camera_index", 0)

        # Calculate delay by determining the ms per frame
        self.delay = 1000 // config.get("camera_frames_per_second", 15)

    def start(self):
        # Start capture and logging
        cap = cv2.VideoCapture(self.camera_index)
        logging.info("Gesture recognition started.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip frame and convert to RGB
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame with mediapipe
            result = self.hands.process(frame_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Extract landmarks and reshape for prediction
                    landmarks = [
                        coord
                        for lm in hand_landmarks.landmark
                        for coord in (lm.x, lm.y, lm.z)
                    ]
                    landmarks = np.array(landmarks).reshape(1, 21, 3)

                    # Make prediction
                    prediction = self.model.predict(landmarks)
                    predicted_label_index = np.argmax(prediction)
                    gesture = self.le.inverse_transform([predicted_label_index])[0]
                    confidence = prediction[0][predicted_label_index]
                    logging.info(f"{gesture} ({confidence:.2f})")

                    # Perform action
                    if confidence >= self.confidence_threshold:
                        self.perform_action(gesture)
                    else:
                        continue

            # Display the frame (FOR TESTING)
            cv2.imshow("Frame", frame)

            # Escape key (FOR TESTING)
            if cv2.waitKey(self.delay) & 0xFF == ord("q"):
                break

            # Delay
            # cv2.waitKey(self.delay)

        cap.release()
        cv2.destroyAllWindows()
        logging.info("Gesture recognition stopped.")

    def perform_action(self, gesture):
        action = self.gesture_action_map.get(gesture)

        # Check if the gesture is one of the volume controls
        if action in ["volumeup", "volumedown"]:
            if self.current_gesture == gesture:
                elapsed_time = time.time() - self.gesture_start_time
                if elapsed_time >= self.gesture_hold_time:
                    # Increase volume change rate based on elapsed time
                    change_rate = min(
                        (elapsed_time - self.gesture_hold_time) * 2, 100
                    )  # Adjust multiplier as needed
                    self.adjust_volume(action, change_rate)
                else:
                    # If not enough time has passed, do nothing or wait
                    return
            else:
                # If gesture changed, start tracking new gesture
                self.current_gesture = gesture
                self.gesture_start_time = time.time()

        else:
            # Perform action with a delay for non-volume gestures
            if action:
                pyautogui.press(action)
                time.sleep(self.gesture_hold_time)  # Delay between actions

    def adjust_volume(self, action, change_rate):
        # Adjust the volume using pyautogui (or any other method) based on the change rate
        step = int(change_rate)
        if action == "volumeup":
            pyautogui.press("volumeup", presses=step)
        elif action == "volumedown":
            pyautogui.press("volumedown", presses=step)


# Runner function (for testing)
if __name__ == "__main__":
    gesture_controller = GestureController("config.json")
    gesture_controller.start()