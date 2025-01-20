"""
Program entry point.
"""

import warnings
from flask import Flask, jsonify, render_template, Response

from modules.controller import Loop

# Clear terminal output and add logging after
warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype()")

app = Flask(__name__)
loop = Loop()


@app.route("/")
def index() -> str:
    """
    Render the HTML page with buttons.
    """

    return render_template("index.html")


@app.route("/start-loop", methods=["POST"])
def start_loop() -> Response:
    """
    Start the background loop.
    """

    response = loop.start_loop()
    print(response)
    return jsonify({"message": response})


@app.route("/stop-loop", methods=["POST"])
def stop_loop() -> Response:
    """
    Stop the background loop.
    """

    response = loop.stop_loop()
    print(response)
    return jsonify({"message": response})


@app.route("/restart-loop", methods=["POST"])
def restart_loop() -> Response:
    """
    Restart the background loop.
    """

    response = loop.restart_loop()
    print(response)
    return jsonify({"message": response})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
