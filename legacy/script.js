async function predictIntent(text) {
    try {
        const modelUrl = "http://" + window.location.host + "/model/model.json";
        const model = await tf.loadLayersModel(modelUrl);
        console.log("Model loaded:");
        console.log(model);
        // TODO: Preprocess the text
        // TODO: Predict the intent
        return
    } catch (error) {
        console.log("Error loading model:");
        console.log(error);
    }
}

async function getIntentResponse(intent) {
  try {
    // TODO: Get the answer for the intent from the db
  } catch (error) {
    console.log("Error getting intent answer:");
    console.log(error);
  }
}

function onAsk() {
    const inputText = document.getElementById("input-text");
    
    if (inputText.value.trim() == "") {
        return
    }

    const messages = document.querySelector(".messages");
    const message = document.createElement("div");
    message.classList.add("message");
    message.classList.add("user-message");
    message.innerText = inputText.value;
    messages.appendChild(message);
    inputText.value = "";

    // TODO: Switch to the chatbot's response
    const responseText = "Hello, I'm a chatbot!";
    // const intent = predictIntent(inputText.value);
    // const responseText = getIntentResponse(intent);
    
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
predictIntent("Hello");

