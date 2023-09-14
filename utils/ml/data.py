import pandas as pd

def get_data():
    questions = pd.read_csv('./data/questions.csv')
    answers = pd.read_csv('./data/answers.csv')
    return questions, answers

def augment_text_data(texts):
    texts_augmented = []

    for text in texts:
        words = text.split(' ')
        n_words = len(words)
        for i in range(n_words):
            for j in range(i+1, n_words):
                texts_augmented.append(' '.join(words[i:j]))

    return texts_augmented
