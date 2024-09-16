function sendPostRequest(url) {
  fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
  })
    .then((response) => response.json())
    .then((data) => {
      document.getElementById("message").innerText = data.message;
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}

function startLoop() {
  sendPostRequest("/start-loop");
}

function stopLoop() {
  sendPostRequest("/stop-loop");
}

function restartLoop() {
  sendPostRequest("/restart-loop");
}
