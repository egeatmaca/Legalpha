import pandas as pd
from sklearn.metrics import classification_report
from legalpha.Legalpha import Legalpha
from legalpha.experiments.LegalphaClf import LegalphaClf
from legalpha.experiments.LegalphaSemSearch import LegalphaSemSearch
from time import time


MODEL_CONSTRUCTORS = {
    'bert-classifier': Legalpha,
    'classifier': LegalphaClf,
    'semantic-search': LegalphaSemSearch
}

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

def test_legalpha(model_name='bert-classifier', test_size=0.2, random_state=42):
    print(f'Test Size: {test_size}')
    print(f'Random State: {random_state}')

    questions_train, questions_test, answers = get_test_data(test_size, random_state)

    X_train = questions_train['text']
    y_train = questions_train['answer_id']

    X_test = questions_test['text']
    y_test = questions_test['answer_id']

    print(f'#Training Questions: {questions_train.shape[0]}')
    print(f'#Test Questions: {questions_test.shape[0]}')
    print(f'#Answers: {answers.shape[0]}')

    model = MODEL_CONSTRUCTORS[model_name]()

    print(f'Training {model_name}...')
    training_start = time()

    if model_name == 'semantic-search':
        model.fit(questions_train, answers)
    else:
        model.fit(X_train, y_train, validation_split=None)
    
    training_end = time()
    print(f'Training Time: {training_end - training_start} seconds')

    print(f'Testing {model_name}...')
    test_start = time()
    
    y_pred = None
    if model_name == 'semantic-search':
        y_pred = predict_all(questions_test, model)
    else:
        y_pred = model.predict(X_test)
    test_results = evaluate(y_test, y_pred)

    test_end = time()
    print(f'Test Time: {test_end - test_start} seconds')
    print(f'Test Results: \n{test_results}')

if __name__ == '__main__':
    test_legalpha()