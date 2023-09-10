import pandas as pd
from sklearn.metrics import classification_report
from Legalpha import Legalpha
from experiments import LegalphaClf, LegalphaBertClf
from time import time

def get_test_data(test_size=0.2, random_state=42):
    questions = pd.read_csv('./data/questions.csv')
    answers = pd.read_csv('./data/answers.csv')

    question_list_train = []
    question_list_test = []
    answer_ids = questions['answer_id'].unique()
    for answer_id in answer_ids:
        questions_answer = questions[questions['answer_id'] == answer_id]

        test_questions = questions_answer.sample(frac=test_size, random_state=random_state)
        
        train_index = questions_answer.index.difference(test_questions.index)
        train_questions = questions_answer.loc[train_index]
        
        question_list_train.append(train_questions)
        question_list_test.append(test_questions)

    questions_train = pd.concat(question_list_train)
    questions_test = pd.concat(question_list_test)

    return questions_train, questions_test, answers

def predict_all(questions_pred, legalpha):
    answers_pred = []
    for question in questions_pred['text']:
        _, answer_id, _ = legalpha.predict(question)
        answers_pred.append(answer_id)
    return answers_pred

def evaluate(answers_true, answers_predicted):
    return classification_report(answers_true, answers_predicted)

def test_legalpha(test_size=0.2, random_state=42, model='semantic_search'):
    print(f'Test Size: {test_size}')
    print(f'Random State: {random_state}')

    legalpha = Legalpha()
    questions_train, questions_test, answers = get_test_data(test_size, random_state)

    print(f'#Training Questions: {questions_train.shape[0]}')
    print(f'#Test Questions: {questions_test.shape[0]}')
    print(f'#Answers: {answers.shape[0]}')

    print('Training Legalpha...')
    training_start = time()
    legalpha.fit(questions_train, answers)
    training_end = time()
    print(f'Training Time: {training_end - training_start} seconds')

    print('Testing Legalpha...')
    answers_test = questions_test['answer_id']
    test_start = time()
    answers_pred = predict_all(questions_test, legalpha)
    test_end = time()
    test_results = evaluate(answers_test, answers_pred)
    print(f'Test Time: {test_end - test_start} seconds')
    print(f'Test Results: \n{test_results}')


def test_legalpha_clf(test_size=0.2, random_state=42, bert_embeddings=False):
    print(f'Test Size: {test_size}')
    print(f'Random State: {random_state}')

    questions_train, questions_test, answers = get_test_data(test_size, random_state)

    print(f'#Training Questions: {questions_train.shape[0]}')
    print(f'#Test Questions: {questions_test.shape[0]}')
    print(f'#Answers: {answers.shape[0]}')

    X_train = questions_train['text']
    y_train = questions_train['answer_id']

    X_test = questions_test['text']
    y_test = questions_test['answer_id']

    legalpha_clf = LegalphaBertClf() if bert_embeddings else LegalphaClf()
    model_name = 'LegalphaBertClf' if bert_embeddings else 'LegalphaClf'

    print(f'Training {model_name}...')
    training_start = time()
    legalpha_clf.fit(X_train, y_train, validation_split=None)
    training_end = time()
    print(f'Training Time: {training_end - training_start} seconds')

    print(f'Testing {model_name}...')
    test_start = time()
    y_pred = legalpha_clf.predict(X_test)
    test_end = time()
    test_results = evaluate(y_test, y_pred)
    print(f'Test Time: {test_end - test_start} seconds')
    print(f'Test Results: \n{test_results}')

if __name__ == '__main__':
    test_legalpha()