import pandas as pd
import os
from models import UserQuestion

USER_QUESTIONS_FOLDER = os.path.join('data', 'user_questions')

def export_user_questions():
    answered_questions = pd.DataFrame(UserQuestion.search({'last_feedback': {'$eq': 1}}))
    unanswered_questions = pd.DataFrame(UserQuestion.search({'last_feedback': {'$in': [0, None]}}))
    
    os.makedirs(USER_QUESTIONS_FOLDER, exist_ok=True)
    answered_questions.to_csv(os.path.join(USER_QUESTIONS_FOLDER, 'answered_questions.csv'), index=False)
    unanswered_questions.to_csv(os.path.join(USER_QUESTIONS_FOLDER, 'unanswered_questions.csv'), index=False)
