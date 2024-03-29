const state = {
  'last_question': '',
  'last_user_question_id': -1,
  'last_answer': '',
  'last_answer_id': -1,
  'retries': 0,
}

const responses_on_positive = [
  "I'm glad I could help! Feel free to ask if you have any other questions.",
  "I'm glad I could be of assistance! Can I help you with anything else?",
  "I'm glad I could be of help! Do you have any other questions?",
  "I'm happy I could help! Let me know if you have any other questions.",
  "I'm happy I could be of assistance! Can I help you with anything else?",
  "I'm happy I could be of help! Do you have any other questions?",
]; 

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
  messages.scroll(0, messages.scrollHeight);
}

async function getAnswer(input) {
  if (input.trim() == "") {
    return;
  }

  let question = encodeURIComponent(input);
  let url = "/answer?question=" + question + "&nth_likely=" + (state.retries + 1)
  if (state.last_user_question_id != -1) {
    url += "&user_question_id=" + state.last_user_question_id;
  }
  const response = await fetch(url);
  const responseJson = JSON.parse(await response.text());

  return responseJson
}

function displayAnswer(answerJson) {
  const topic = answerJson.topic;
  const topicExists = topic != undefined && topic != null && topic != "";
  const answerFormatted = formatAnswer(answerJson.answer);

  document.querySelectorAll(".feedback-container").forEach((e) => e.remove());

  const responseMessage = document.createElement("div");
  responseMessage.classList.add("message");
  responseMessage.classList.add("bot-message");

  const answerContainer = document.createElement("div");
  answerContainer.classList.add("answer-container");
  if (topicExists) {
    const topicParagraph = document.createElement("p");
    topicParagraph.classList.add("topic");
    topicParagraph.innerText = "Topic: " + topic;
    answerContainer.appendChild(topicParagraph);
  }
  const answerParagraph = document.createElement("span");
  answerParagraph.classList.add("answer");
  answerParagraph.innerHTML = answerFormatted;
  answerContainer.appendChild(answerParagraph);
  responseMessage.appendChild(answerContainer);
  
  if (topicExists) {
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
  
  messages.scroll(0, messages.scrollHeight);
}

function formatAnswer(answer) {
  if (answer.startsWith('"')) {
    answer = answer.substring(1, answer.length);
  }

  if (answer.endsWith('"')) {
    answer = answer.substring(0, answer.length - 1);
  }

  while (answer.includes("\\n")) {
    answer = answer.replace("\\n", "<br>");
  }

  while (answer.includes('\\"')) {
    answer = answer.replace('\\"', '"');
  }

  return answer;
}

async function onAsk() {
  const input = getInput();
  displayInput(input);

  state.last_question = input;
  state.last_user_question_id = -1;
  state.retries = 0;

  const answerJson = await getAnswer(input);
  displayAnswer(answerJson);

  state.last_user_question_id = answerJson.user_question_id;
  state.last_answer = answerJson.answer;
  state.last_answer_id = answerJson.answer_id;
}

async function onNegativeFeedback() {
  displayInput('No, I was looking for something else.')

  state.retries = state.retries + 1;

  const answerJson = await getAnswer(state.last_question);
  displayAnswer(answerJson);

  await fetch(
    "/handle_feedback?user_question_id=" + state.last_user_question_id + "&answer_id=" + state.last_answer_id + "&feedback=0",
    {method: 'PUT'}
  );

  state.last_answer = answerJson.answer;
  state.last_answer_id = answerJson.answer_id;
}

async function onPositiveFeedback() {
  const randomIndex = Math.floor(Math.random() * responses_on_positive.length);
  const answerJson = { 'answer': responses_on_positive[randomIndex] };
  displayAnswer(answerJson);

  await fetch(
    "/handle_feedback?user_question_id=" + state.last_user_question_id + "&answer_id=" + state.last_answer_id + "&feedback=1",
    {method: 'PUT'}
  );

  state.last_question = "";
  state.last_user_question_id = -1;
  state.last_answer = "";
  state.last_answer_id = -1;
  state.retries = 0;
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
