import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from .prediction import Prediction


class Model:
    """
    Class for interfacing with TF model.
    """

    def __init__(self, model_path: str, label_path: str) -> None:
        """
        Creates an instance of the model.
        """

        # Load interpreter
        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Load label encoder classes
        le_classes = np.load(label_path, allow_pickle=True)
        self.le = LabelEncoder()
        self.le.fit(le_classes)

        # Load mediapipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def make_prediction(self, frame: np.ndarray) -> Prediction:
        """
        Makes a prediction from a frame.
        """

        # Process with mediapipe
        result = self.hands.process(frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Extract landmarks and reshape for prediction
                landmarks = [
                    coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)
                ]
                landmarks = np.array(landmarks).reshape(1, 21, 3).astype(np.float32)

                # Set tensor
                self.interpreter.set_tensor(self.input_details[0]["index"], landmarks)

                # Invoke interpreter
                self.interpreter.invoke()

                # Get prediction
                prediction = self.interpreter.get_tensor(self.output_details[0]["index"])
                predicted_label_index = np.argmax(prediction)
                gesture = self.le.inverse_transform([predicted_label_index])[0]
                confidence = prediction[0][predicted_label_index]

                return Prediction(gesture=gesture, confidence=confidence)

        # Return None if no hand landmarks are detected.
        return None
