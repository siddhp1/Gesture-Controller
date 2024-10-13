import numpy as np


class Prediction:
    """
    Model prediction result.
    """

    def __init__(self, gesture: str, confidence: np.float32) -> None:
        """
        Initialize a prediction instance.
        """
        # Checks that gesture is of type string
        if not isinstance(gesture, str):
            raise ValueError("Gesture must be a string.")

        # Checks that confidence is a float and between 0 and 1
        if not isinstance(confidence, np.float32) or not (0.0 <= confidence <= 1.0):
            raise ValueError("Confidence must be a float between 0 and 1.")

        self.gesture = gesture
        self.confidence = confidence
