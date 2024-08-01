import os
import signal
import click
from multiprocessing import Process
from gesture_controller import GestureController

# CONSTANTS
CONFIG_PATH = "config.json"
PROCESS_PATH = "process.pid"


def run_gesture_controller(config_path):
    gesture_controller = GestureController(config_path)
    gesture_controller.start()


@click.group()
def cli():
    pass


@cli.command()
def start():
    # Check if process is already running
    # if os.path.exists(PROCESS_PATH):
    #     click.echo("Gesture recognition is already running.")
    #     return

    # Create and start process
    process = Process(target=run_gesture_controller, args=(CONFIG_PATH,))
    process.start()

    # Save process id to a file
    with open(PROCESS_PATH, "w") as f:
        f.write(str(process.pid))

    click.echo("Gesture recognition started.")


@cli.command()
def stop():
    # Look for process file and terminate process
    if os.path.exists(PROCESS_PATH):
        with open(PROCESS_PATH, "r") as f:
            pid = int(f.read())
            print(pid)

        if os.name == "nt":  # Windows
            os.kill(pid, signal.CTRL_BREAK_EVENT)
        else:  # Unix
            os.kill(pid, signal.SIGTERM)

        # Delete process file
        os.remove(PROCESS_PATH)
        click.echo("Gesture recognition stopped.")
    else:
        click.echo("No running process found.")


@cli.command()
def restart():
    if os.path.exists(PROCESS_PATH):
        stop()
    start()
    click.echo("Gesture recognition restarted.")


@cli.command()
def status():
    # Find process file
    if os.path.exists(PROCESS_PATH):
        # Read process id and stop
        with open(PROCESS_PATH, "r") as f:
            pid = int(f.read())

        # Attempt to reach process
        try:
            os.kill(pid, 0)
            click.echo("Gesture recognition is running.")
        except OSError:
            click.echo("Gesture recognition is not running.")
    else:
        click.echo("No running process found.")


if __name__ == "__main__":
    cli()
