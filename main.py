from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from LegalphaQA import Legalpha
from inject_data import inject_data

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
    answer = legalpha.answer(input_question)
    return answer


if __name__ == '__main__':
    inject_data()
    uvicorn.run(app, host='0.0.0.0', port=8000)