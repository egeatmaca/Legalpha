import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from time import time
from utils.data import get_data
from utils.ml import MODEL_CONSTRUCTORS

def stratified_train_test_split(questions, test_size=0.2, random_state=42, shuffle=True):
    question_list_train = []
    question_list_test = []
    answer_ids = questions['answer_id'].unique()
    for answer_id in answer_ids:
        questions_answer = questions[questions['answer_id'] == answer_id]

        if shuffle:
            questions_answer = questions_answer.sample(frac=1, random_state=random_state)

        test_questions = questions_answer.sample(frac=test_size, random_state=random_state)
        
        train_index = questions_answer.index.difference(test_questions.index)
        train_questions = questions_answer.loc[train_index]
        
        question_list_train.append(train_questions)
        question_list_test.append(test_questions)

    questions_train = pd.concat(question_list_train)
    questions_test = pd.concat(question_list_test)

    X_train = questions_train['text']
    X_test = questions_test['text']
    y_train = questions_train['answer_id']
    y_test = questions_test['answer_id']

    return X_train, X_test, y_train, y_test

def predict_all(questions_pred, legalpha):
    answers_pred = []
    for question in questions_pred['text']:
        _, answer_id, _ = legalpha.predict(question)
        answers_pred.append(answer_id)
    return answers_pred

def evaluate(answers_true, answers_predicted):
    return classification_report(answers_true, answers_predicted)

def test_legalpha(model_name='bert-embedding-classifier', test_size=0.2, sampling='random', random_state=42):
    if model_name not in MODEL_CONSTRUCTORS.keys():
        raise ValueError(f'Invalid model name: {model_name}')

    print(f'Test Size: {test_size}')
    print(f'Sampling Method: {sampling}')
    print(f'Random State: {random_state}')

    questions, answers = get_data()
    if sampling == 'random':
        X = questions['text']
        y = questions['answer_id']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    elif sampling == 'stratified':
        X_train, X_test, y_train, y_test = stratified_train_test_split(questions, test_size=test_size, random_state=random_state)
    else:
        raise ValueError(f'Invalid sampling method: {sampling}')

    print(f'#Training Questions: {X_train.shape[0]}')
    print(f'#Test Questions: {X_test.shape[0]}')
    print(f'#Answers: {answers.shape[0]}')

    model = MODEL_CONSTRUCTORS[model_name]()

    print(f'Training {model_name}...')
    training_start = time()

    if model_name == 'semantic-search':
        questions_train = pd.concat([X_train, y_train], axis=1)
        model.fit(questions_train, answers)
    else:
        model.fit(X_train, y_train)
    
    training_end = time()
    print(f'Training Time: {training_end - training_start} seconds')

    print(f'Testing {model_name}...')
    test_start = time()
    
    y_pred = None
    if model_name == 'semantic-search':
        questions_test = pd.concat([X_test, y_test], axis=1)
        y_pred = predict_all(questions_test, model)
    else:
        y_pred = model.predict(X_test)
    test_results = evaluate(y_test, y_pred)

    test_end = time()
    print(f'Test Time: {test_end - test_start} seconds')
    print(f'Test Results: \n{test_results}')

if __name__ == '__main__':
    test_legalpha()