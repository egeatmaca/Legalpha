from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from typing import Annotated
from sklearn.metrics.pairwise import cosine_similarity
from simpletransformers.language_representation import RepresentationModel
from models import Question, Answer
from inject_data import inject_data

# Inject data into MongoDB
inject_data()

# Load BERT model
bert = RepresentationModel('bert', 'bert-base-cased', use_cuda=False)

# Initialize FastAPI
app = FastAPI()
templates = Jinja2Templates(directory='templates')
static_files = StaticFiles(directory='static')
app.mount('/static', static_files, name='static')


@app.get('/')
def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


@app.get('/answer')
def answer(prompt: Annotated[str, Form()]):
    prompt_embedding = bert.encode_sentences(
        [prompt], combine_strategy='mean')[0]

    questions = Question.objects.all()
    max_similarity = -1
    most_similar_question = None
    for question in questions:
        if not question.embedding:
            question.embedding = bert.encode_sentences(
                [question.text], combine_strategy='mean')[0]
            question.update()

        similarity = cosine_similarity(
            prompt_embedding, question.embedding)[0][0]

        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_question = question

    answer = Answer.objects.get_by_pk(most_similar_question.answer_id).text

    return answer

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)