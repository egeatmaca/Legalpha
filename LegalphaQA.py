from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from models import Question, Answer

class Legalpha:
    def __init__(self):
        if self.model_path_exists():
            self.load_model()
        else:
            self.model = QuestionAnsweringModel('bert', 'bert-base-cased', use_cuda=False)

        self.train_data = []
        self.eval_data = []
        
        questions = Question.search({'answer_id': {'$exists': True}})
        for question in questions:
            answer = None
            answers = Answer.search({'id': question.get('answer_id')})
            for a in answers:
                answer = a
                break

            if not answer:
                continue

            data = {
                'context': answer.get('text'),
                'qas': [{
                    'id': question.get('id'),
                    'question': question.get('text'),
                    'answers': [{
                        'text': answer.get('text'),
                        'answer_start': 0
                    }]
                }]
            }

            if np.random.rand() < 0.8:
                self.train_data.append(data)
            else:
                self.eval_data.append(data)

    def model_path_exists(self) -> bool:
        '''
        This function checks if the data path exists.
        '''

        return os.path.exists('qa_model/')

    def train(self):
        '''
        This function trains the model.
        '''

        self.model.train_model(self.train_data)

    def evaluate(self):
        '''
        This function evaluates the model.
        '''

        result, model_outputs, predictions = self.model.eval_model(self.eval_data)

        return result
    
    def predict(self, question: str) -> str:
        '''
        @param question: The question to answer
        @return answer: The answer
        '''

        prediction = self.model.predict([question])

        return prediction


    def save_model(self, path: str = 'qa_model/'):
        '''
        This function saves the model.
        '''

        self.model.save_model(path)
    
    def load_model(self, path: str = 'qa_model/'):
        '''
        This function loads the model.
        '''

        self.model = QuestionAnsweringModel('bert', path)

    def answer(self, input_question: str) -> str:
        if not os.path.exists('qa_model/'):
            self.train()
            self.save_model()

        answer = self.predict(input_question)

        print(f'Input question: {input_question}')
        print(f'Answer: {answer}')

        return answer

if __name__ == '__main__':
    legalpha = Legalpha()
    legalpha.train()
    legalpha.save_model()
    metrics = legalpha.evaluate()
    print(metrics)