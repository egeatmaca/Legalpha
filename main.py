from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from Legalpha import Legalpha
from inject_data import inject_data
import numpy as np

# Initialize FastAPI
app = FastAPI()
templates = Jinja2Templates(directory='templates')
static_files = StaticFiles(directory='static')
app.mount('/static', static_files, name='static')

# Initialize Legalpha
legalpha = Legalpha()

# Initialize response templates
response_templates = [{'question_pointer': 'As far as I understood, you asked: ', 
                       'answer_pointer': 'According to my knowledge, '},
                       {'question_pointer': 'I think you asked: ',
                        'answer_pointer': 'I heard from my researcher friends,'},]


# Routes
@app.get('/')
def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.get('/answer')
def answer(request: Request):
    input_question = request.query_params['question']
    answer, matched_question = legalpha.answer(input_question)
    answer = answer[0].lower() + answer[1:]
    random_template = np.random.choice(response_templates)
    answer = random_template['question_pointer'] + matched_question + random_template['answer_pointer'] +  answer
    return answer


if __name__ == '__main__':
    inject_data()
    uvicorn.run(app, host='0.0.0.0', port=8000)