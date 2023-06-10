import numpy as np
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from Legalpha import Legalpha
from inject_data import inject_data
from models import Question

# Initialize FastAPI
app = FastAPI()
templates = Jinja2Templates(directory='templates')
static_files = StaticFiles(directory='static')
app.mount('/static', static_files, name='static')

# Initialize Legalpha
legalpha = Legalpha()

# Initialize response templates
response_templates = [{'question_pointer': 'As far as I understood, you asked:', 
                       'answer_pointer': 'According to my knowledge,'},
                       {'question_pointer': 'I think you asked:',
                        'answer_pointer': 'I heard from my researcher friends,'},
                        {'question_pointer': 'I believe you asked:',
                         'answer_pointer': 'You should know,'},
                        {'question_pointer': 'So your question is:',
                            'answer_pointer': 'By reading a lot through the internet, I found out, that',
                         },]

alternative_response_templates = [{'question_pointer': 'If that was not your question, you might also be asking:',
                                   'answer_pointer': 'I think you should know that,'},
                                    {'question_pointer': 'Okay, then let me try again. Do you mean:',
                                        'answer_pointer': 'I think that would help if you knew,'},
                                    {'question_pointer': 'Sorry to hear that, I can try again. Perhaps you were asking:', 
                                     'answer_pointer': 'My answer would be,'},
                                    {'question_pointer': 'In a second thought, I think you were asking:',
                                        'answer_pointer': 'In this context, '},]                              


# Routes
@app.get('/')
def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.get('/answer')
def answer(request: Request):
    question = request.query_params['question']

    nth_similar = 1
    if 'nth_similar' in request.query_params.keys():
        nth_similar = int(request.query_params['nth_similar'])
    
    template = None
    if nth_similar == 1:
        template = np.random.choice(response_templates)
    else:
        template = np.random.choice(alternative_response_templates)

    answer, matched_question = legalpha.answer(question, nth_similar=nth_similar)

    if answer and matched_question:
        answer = answer[0].lower() + answer[1:]
        answer = template['question_pointer'] + ' "' + matched_question + '" ' + template['answer_pointer'] + ' ' + answer
    else:
        Question(text=question).create()
        answer = 'I am sorry, I could not find an answer to your question.'

    print('Question: ', question)
    print('Nth Similar: ', nth_similar)
    print('Answer: ', answer)
    return answer


if __name__ == '__main__':
    inject_data()
    uvicorn.run(app, host='0.0.0.0', port=8000)