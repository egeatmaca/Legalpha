import pandas as pd

def read_csv(path):
    seperators = [';', ',', '|', '\t']
    for sep in seperators:
        try:
            return pd.read_csv(path, sep=sep)
        except:
            continue

def get_data():
    questions = read_csv('./data/questions.csv')
    answers = read_csv('./data/answers.csv')
    return questions, answers
