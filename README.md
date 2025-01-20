# Gesture Controller

Gesture Controller is a cross-platform application for users to control media functions using hand gestures, with real-time camera capture and high-performance inference for responsive user actions. Built with Flask, Tensorflow, OpenCV and Mediapipe.

Features a dense neural networked trained to 97% validation accuracy on landmark data collected from the [HaGRID (512px) dataset](https://github.com/hukenovs/hagrid).

## Installation

1. **Clone the repository:**
    ```bash
    $ git clone https://github.com/siddhp1/Gesture-Controller.git
    $ cd Gesture-Controller/app
    ```

2. **Create environment and install dependencies:**

    ```bash
    $ python3 -m venv venv # Use Python 3.12
    $ source venv/bin/activate # On Windows use `venv\Scripts\activate`
    $ pip install -r requirements.txt
    ```

## Usage

1. **Run application**

    ```bash
    $ python -m main
    ```

2. **Open GUI**

    Go to `http://localhost:5000` in your web browser.

# License

This project is licensed under the MIT License.