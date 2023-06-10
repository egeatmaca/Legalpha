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
  document.querySelectorAll(".feedback-container").forEach((e) => e.remove());

  const responseMessage = document.createElement("div");
  responseMessage.classList.add("message");
  responseMessage.classList.add("bot-message");

  const responseText = document.createElement("p");
  responseText.innerHTML = responseTextFormatted;
  responseMessage.appendChild(responseText);
  
  if (!responseTextFormatted.includes("I could not find an answer")) {
    const thumbsUpButton = document.createElement("button");
    thumbsUpButton.setAttribute("id", "thumbs-up-button");
    thumbsUpButton.classList.add("feedback-button");
    thumbsUpButton.innerText = "👍";
    thumbsUpButton.addEventListener("click", onPositiveFeedback);

    const thumbsDownButton = document.createElement("button");
    thumbsDownButton.setAttribute("id", "thumbs-down-button");
    thumbsDownButton.classList.add("feedback-button");
    thumbsDownButton.innerText = "👎";
    thumbsDownButton.addEventListener("click", onNegativeFeedback);

    const feedbackContainer = document.createElement("div");
    feedbackContainer.classList.add("feedback-container");
    feedbackContainer.appendChild(thumbsUpButton);
    feedbackContainer.appendChild(thumbsDownButton);

    responseMessage.appendChild(feedbackContainer);
  }

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

  while (responseText.includes('\\"')) {
    responseText = responseText.replace('\\"', '"');
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

async function onNegativeFeedback() {
  state.retries = state.retries + 1;
  displayInput('No, I was looking for something else.')
  const answer = await getAnswer(state.last_input);
  displayAnswer(answer);
}

async function onPositiveFeedback() {
  // TODO: Implement
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
}


document.addEventListener("DOMContentLoaded", initialize);
