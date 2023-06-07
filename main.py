from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from Legalpha import Legalpha
from inject_data import inject_data
import json

# Initialize FastAPI
app = FastAPI()
templates = Jinja2Templates(directory='templates')
static_files = StaticFiles(directory='static')
app.mount('/static', static_files, name='static')

# Initialize Legalpha
legalpha = Legalpha()

# Routes
@app.get('/')
def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.get('/answer')
def answer(request: Request):
    input_question = request.query_params['question']
    answer, matched_question = legalpha.answer(input_question)
    answer = answer[0].lower() + answer[1:]
    answer = f'As far as I understood, you asked: {matched_question} According to my knowledge, {answer}'
    return answer


if __name__ == '__main__':
    inject_data()
    uvicorn.run(app, host='0.0.0.0', port=8000)