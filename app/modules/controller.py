import threading
from modules.camera import Camera
from modules.command import Command
from modules.config import Config
from modules.model import Model


class Loop:
    """
    Class to manage the loop.
    """

    def __init__(self) -> None:
        """
        Initialize.
        """
        self.loop_active = False
        self.event = threading.Event()
        self.thread = None

    def start_loop(self) -> str:
        """
        Function to start the loop.
        """
        if self.thread is None or not self.thread.is_alive():
            self.loop_active = True
            self.thread = threading.Thread(target=self.background_loop, args=(self.event,))
            self.thread.start()
            return "Controller started!"
        else:
            return "Controller is already running!"

    def stop_loop(self) -> str:
        """
        Function to stop the loop.
        """
        if self.loop_active:
            self.loop_active = False
            self.event.set()
            self.thread.join()

            # Set flag back to false
            self.event.clear()
            return "Controller stopped!"
        else:
            return "Controller is not running!"

    def restart_loop(self) -> None:
        """
        Function to restart the loop.
        """
        self.stop_loop()
        return self.start_loop()

    def background_loop(self, stop: threading.Event) -> None:
        """
        Loop functionality.
        """
        config = Config(config_path="config.json")
        model = Model(
            model_path=config.get_value("model_path"), label_path=config.get_value("label_path")
        )
        camera = Camera(
            config.get_value("camera_index"), 1000 // config.get_value("camera_frames_per_second")
        )
        command = Command(
            config.get_value("gesture_action_map"), config.get_value("gesture_hold_time")
        )

        confidence_threshold = config.get_value("confidence_threshold")

        try:
            while camera.is_opened():
                pred = model.make_prediction(camera.read_frame()[1])

                # Perform action
                if pred:
                    if pred.confidence >= confidence_threshold:
                        command.perform_action(pred.gesture)

                # Check for flag
                if stop.is_set():
                    break

                camera.wait()
        finally:
            camera.release()
