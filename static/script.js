async function onAsk() {
    const inputText = document.getElementById("input-text");
    const input = inputText.value;
    inputText.value = "";
    
    if (input.trim() == "") {
      return;
    }

    const messages = document.querySelector(".messages");
    const message = document.createElement("div");
    message.classList.add("message");
    message.classList.add("user-message");
    message.innerText = input;
    messages.appendChild(message);

    const response = await fetch("/answer?question="+input)
    const responseText = await response.text();
    
    const responseMessage = document.createElement("div");
    responseMessage.classList.add("message");
    responseMessage.innerText = responseText;
    messages.appendChild(responseMessage);

    window.scrollTo(0, document.body.scrollHeight);
}

function initialize() {
    document.getElementById("ask-button").addEventListener("click", onAsk);
    document.getElementById("input-text").addEventListener("keypress", function(e) {
        if (e.key === "Enter") {
            onAsk();
        }
    });
}

document.addEventListener("DOMContentLoaded", initialize);

