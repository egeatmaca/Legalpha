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
def answer(request: Request):
    input_question = request.query_params['question']
    input_embedding = bert.encode_sentences([input_question], combine_strategy='mean')

    questions = Question.search({'_id': {'$exists': True}})
    max_similarity = -1
    most_similar_question = None
    for question in questions:
        if 'embedding' not in question:
            question['embedding'] = bert.encode_sentences([question.text], combine_strategy='mean')
            Question(**question).update()

        similarity = cosine_similarity(input_embedding, question.get('embedding'))[0][0]

        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_question = question

    answer = Answer.search({'id': most_similar_question['answer_id']}).next().get('text')

    return answer

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)