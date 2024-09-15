"""
Program entry point.
"""

import warnings
from modules.camera import Camera
from modules.command import Command
from modules.config import Config
from modules.model import Model

# Clear terminal output and add logging after
warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype()")


def main() -> None:
    """
    Runs the program.
    """

    config = Config(config_path="config.json")
    model = Model(
        model_path=config.get_value("model_path"), label_path=config.get_value("label_path")
    )
    camera = Camera(
        config.get_value("camera_index"), 1000 // config.get_value("camera_frames_per_second")
    )
    command = Command(config.get_value("gesture_action_map"), config.get_value("gesture_hold_time"))

    confidence_threshold = config.get_value("confidence_threshold")

    try:
        while camera.is_opened():
            pred = model.make_prediction(camera.read_frame()[1])

            # Perform action
            if pred:
                if pred.confidence >= confidence_threshold:
                    command.perform_action(pred.gesture)

            camera.wait()
    finally:
        camera.release()


if __name__ == "__main__":
    main()
