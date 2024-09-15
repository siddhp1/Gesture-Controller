import cv2
import numpy as np


class Camera:
    """
    Class to handle camera operations.
    """

    def __init__(self, camera_index: int, delay: int) -> None:
        """
        Initialize the camera.
        """

        self.camera_index = camera_index
        self.delay = delay
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise ValueError(f"Camera with index {self.camera_index} cannot be opened.")

    def is_opened(self) -> bool:
        """
        Checks if the camera is opened.
        """

        return self.cap.isOpened()

    def read_frame(self) -> "tuple[bool, np.ndarray]":
        """
        Read a frame from the capture.
        """

        ret, frame = self.cap.read()
        if ret:
            # Flip frame horizontally
            frame = cv2.flip(frame, 1)

            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            return ret, frame_rgb
        return ret, None

    def release(self) -> None:
        """
        Release the camera.
        """

        self.cap.release()
        cv2.destroyAllWindows()

    def wait(self) -> None:
        """
        Wait for the specified delay.
        """

        cv2.waitKey(self.delay)
