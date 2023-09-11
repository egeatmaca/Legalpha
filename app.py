import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

from legalpha.Legalpha import Legalpha
from models import Answer
from jobs.inject_data import inject_data
from jobs.train_legalpha import train_legalpha
import utils.feedback

# Initialize FastAPI
app = FastAPI()
templates = Jinja2Templates(directory='templates')
static_files = StaticFiles(directory='static')
app.mount('/static', static_files, name='static')

# Initialize Legalpha
legalpha = Legalpha()

# Initialize response templates

response_templates = ['According to my knowledge,', 
                      'I heard from my researcher friends,', 
                      'By reading a lot through the internet, I found out, that,',
                      'I am not a lawyer, but I think, that,',
                      'Some websites I have read, say,',
                      'To answer your question, I have found the following:',
                      'I have found the following:',
                      'I have found the following answer regarding your question:']

alternative_response_templates = ['Sorry to hear that, let me try again.', 
                                  'Sorry to hear that, I can try again.', 
                                  'Oops, let me try again.', 
                                  'Oops, trying again...', 
                                  'In a second thought, you might be looking for this:',
                                  'My next answer would be:']   

all_templates = response_templates + alternative_response_templates

# Routes
@app.get('/')
def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.get('/answer')
def answer(request: Request):
    # Get parameters
    try:
        question = request.query_params['question']

        user_question_id = None
        if 'user_question_id' in request.query_params.keys():
            user_question_id = int(request.query_params['user_question_id'])

        nth_likely = 1
        if 'nth_likely' in request.query_params.keys():
            nth_likely = int(request.query_params['nth_likely'])
    except:
        return Response(status_code=400, content='Invalid parameters.')

    # Choose a random response template
    template = None
    if nth_likely == 1:
        template = np.random.choice(response_templates)
    else:
        template = np.random.choice(alternative_response_templates)

    # Predict answer
    answer_id = legalpha.predict_nth_likely([question], nth_likely)[0]
    answer_id = int(answer_id) if answer_id is not None else None

    # Get answer text
    answer = None
    if answer_id is not None:
        answer = Answer(id=answer_id).read()['text']
        answer = answer[0].lower() + answer[1:] if template[-1] == ',' else answer
        answer = template + ' ' + answer
    else:
        answer = 'I am sorry, I could not find an answer to your question.'

    # Create a user question, if not exists
    if user_question_id is None:
        user_question = utils.feedback.create_user_question(question)
        user_question_id = user_question.id
        
    # Set the last answer of user question
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
    # Inject data
    inject_data()

    # Train Legalpha, if no saved model exists
    legalpha_pretrained_path = os.environ.get('LEGALPHA_PRETRAINED_PATH')
    legalpha_questions_path = os.environ.get('LEGALPHA_QUESTIONS_PATH')
    if not os.path.exists(legalpha_pretrained_path) and os.path.exists(legalpha_questions_path):
        train_legalpha(legalpha_pretrained_path)

    # Load Legalpha
    legalpha.load(legalpha_pretrained_path)
    
    # Run the app
    uvicorn.run(app, host='0.0.0.0', port=8000)

if __name__ == '__main__':
    run_app()