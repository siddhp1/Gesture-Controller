import pyautogui
import time


class Command:
    """
    Command management.
    """

    def __init__(self, gesture_action_map: dict, gesture_hold_time: float) -> None:
        """
        Creates an instance of the command manager.
        """

        self.gesture_action_map = gesture_action_map
        self.gesture_hold_time = gesture_hold_time
        self.current_gesture = None
        self.gesture_start_time = None

    def perform_action(self, gesture: str) -> None:
        """
        Performs an action.
        """

        action = self.gesture_action_map.get(gesture)

        # Check if the gesture is one of the volume controls
        if action in ["volumeup", "volumedown"]:
            if self.current_gesture == gesture:
                elapsed_time = time.time() - self.gesture_start_time
                if elapsed_time >= self.gesture_hold_time:
                    # Increase volume change rate based on elapsed time
                    change_rate = min(
                        (elapsed_time - self.gesture_hold_time) * 1.2, 100
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

                # this wil need to be changed to work universally
                time.sleep(self.gesture_hold_time)  # Delay between actions

    def adjust_volume(self, action: str, change_rate: float) -> None:
        """
        Adjusts volume.
        """

        step = int(change_rate)
        if action == "volumeup":
            pyautogui.press("volumeup", presses=step)
        elif action == "volumedown":
            pyautogui.press("volumedown", presses=step)
