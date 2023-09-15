import pandas as pd
import os

DF_COLUMNS = {
    'questions': pd.Series(['id', 'text', 'answer_id']),
    'answers': pd.Series(['id', 'text', 'topic'])
}

def read_csv(path):
    file_name = os.path.basename(path).replace('.csv', '')
    seperators = [';', ',', '|', '\t']
    
    for sep in seperators:
        try:
            df = pd.read_csv(path, sep=sep)
            if file_name in DF_COLUMNS.keys():
                assert DF_COLUMNS[file_name].isin(df.columns).all()
            return df
        except:
            continue

def get_data():
    questions = read_csv('./data/questions.csv')
    answers = read_csv('./data/answers.csv')
    return questions, answers
