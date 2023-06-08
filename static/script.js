let state = {
  'last_input': '',
  'retries': 0,
}

function getInput() {
  const inputElement = document.getElementById("input-text");
  const input = inputElement.value;
  inputElement.value = "";
  return input;
}

function displayInput(input) {
  if (input.trim() == "") {
    return;
  }

  const messages = document.querySelector(".messages");
  const message = document.createElement("div");
  message.classList.add("message");
  message.classList.add("user-message");
  message.innerText = input;
  messages.appendChild(message);

  window.scrollTo(0, document.body.scrollHeight);
}

async function getAnswer(input) {
  if (input.trim() == "") {
    return;
  }

  let question = encodeURIComponent(input);
  const response = await fetch(
    "/answer?question=" + question + "&nth_similar=" + (state.retries + 1)
  );
  const responseText = await response.text();
  const responseTextFormatted = formatResponseText(responseText);

  return responseTextFormatted;
}

function displayAnswer(responseTextFormatted) {
  const responseMessage = document.createElement("div");
  responseMessage.classList.add("message");
  responseMessage.innerHTML = responseTextFormatted;

  const messages = document.querySelector(".messages");
  messages.appendChild(responseMessage);

  window.scrollTo(0, document.body.scrollHeight);
}

function formatResponseText(responseText) {
  if (responseText.startsWith('"')) {
    responseText = responseText.substring(1, responseText.length);
  }

  if (responseText.endsWith('"')) {
    responseText = responseText.substring(0, responseText.length - 1);
  }

  while (responseText.includes("\\n")) {
    responseText = responseText.replace("\\n", "<br>");
  }

  return responseText;
}

async function onAsk() {
  const input = getInput();
  displayInput(input);
  const answer = await getAnswer(input);
  displayAnswer(answer);
  state.last_input = input;
  state.retries = 0;
}

async function onRetry() {
  displayInput('No, I was looking for something else.')
  state.retries = state.retries + 1;
  const answer = await getAnswer(state.last_input);
  displayAnswer(answer);
}

function initialize() {
  document.getElementById("ask-button").addEventListener("click", onAsk);

  document
    .getElementById("input-text")
    .addEventListener("keypress", function (e) {
      if (e.key === "Enter") {
        onAsk();
      }
    });

  document.getElementById("retry-button").addEventListener("click", onRetry);
}


document.addEventListener("DOMContentLoaded", initialize);
