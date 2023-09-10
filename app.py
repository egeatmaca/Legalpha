import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

from legalpha.LegalphaSemSearch import LegalphaSemSearch as Legalpha
from jobs.inject_data import inject_data
import utils.feedback

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

all_templates = response_templates + alternative_response_templates
all_answer_pointers = [template['answer_pointer'] for template in all_templates]

# Routes
@app.get('/')
def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.get('/answer')
def answer(request: Request):
    try:
        question = request.query_params['question']

        user_question_id = None
        if 'user_question_id' in request.query_params.keys():
            user_question_id = int(request.query_params['user_question_id'])

        nth_similar = 1
        if 'nth_similar' in request.query_params.keys():
            nth_similar = int(request.query_params['nth_similar'])
    except:
        return Response(status_code=400, content='Invalid parameters.')

    template = None
    if nth_similar == 1:
        template = np.random.choice(response_templates)
    else:
        template = np.random.choice(alternative_response_templates)

    answer, answer_id, matched_question = legalpha.predict(question, nth_similar=nth_similar)
    answer_id = int(answer_id) if answer_id is not None else None

    if answer and matched_question:
        answer = answer[0].lower() + answer[1:]
        answer = template['question_pointer'] + ' "' + matched_question + '" ' + template['answer_pointer'] + ' ' + answer
    else:
        answer = 'I am sorry, I could not find an answer to your question.'

    if user_question_id is None:
        user_question = utils.feedback.create_user_question(question)
        user_question_id = user_question.id
        
    utils.feedback.set_last_answer(user_question_id, answer_id)

    return {'answer': answer, 'answer_id': answer_id, 'user_question_id': user_question_id}

@app.put('/handle_feedback')
def handle_feedback(request: Request):
    try:
        user_question_id = int(request.query_params['user_question_id'])
        answer_id = int(request.query_params['answer_id'])
        feedback = int(request.query_params['feedback'])
    except:
        return Response(status_code=400, content='Invalid parameters.')

    user_question_updated = utils.feedback.handle_feedback(
        user_question_id=user_question_id,
        answer_id=answer_id,
        feedback=feedback
    )

    if not user_question_updated:
        return 'No question found to update.'
    
    return 'Feedback received.'
    

@app.get('/about')
def about(request: Request):
    return templates.TemplateResponse('about.html', {'request': request})

def run_app():
    inject_data()
    legalpha.fit()
    uvicorn.run(app, host='0.0.0.0', port=8000)

if __name__ == '__main__':
    run_app()