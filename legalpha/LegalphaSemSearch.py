import pandas as pd
from simpletransformers.language_representation import RepresentationModel
from sklearn.metrics.pairwise import cosine_similarity
from models import Question, Answer

class LegalphaSemSearch:
    bert = RepresentationModel('bert', 'bert-base-uncased', use_cuda=False)

    def calculate_sentence_embedding(self, sentence: str) -> list:
        '''
        @param sentence: The sentence to embed
        @return: The embedding of the sentence

        This function takes in a sentence and returns the embedding of the sentence.
        '''

        return self.bert.encode_sentences([sentence], combine_strategy='mean').tolist()
    
    def calculate_sentence_embedding_db(self, question: pd.Series) -> list:
        '''
        @param sentence: The sentence to embed
        @return: The embedding of the sentence

        This function takes in a sentence and returns the embedding of the sentence.
        '''

        question = question.to_dict()
        embedding = question.get('embedding')

        if embedding:
            return embedding
        else:
            sentence = question.get('text')
            embedding = self.bert.encode_sentences(sentence, combine_strategy='mean').tolist()
            question['embedding'] = embedding
            Question(**question).update()
            return embedding
    
    def calculate_cosine_similarity(self, embedding1: str, embedding2: str) -> float:
        '''
        @param embedding1: The first embedding
        @param embedding2: The second embedding
        @return: The cosine similarity of the two embeddings

        This function takes in two embeddings and returns the cosine similarity of the two embeddings.
        '''

        return cosine_similarity(embedding1, embedding2)[0][0]
    
    def fit(self, questions: pd.DataFrame=None, answers: pd.DataFrame=None):
        if questions is not None and answers is not None:
            # Check if questions and answers of correct type
            if not isinstance(questions, pd.DataFrame) or not isinstance(answers, pd.DataFrame):
                raise TypeError('Questions and answers should be of type pandas.DataFrame')

            # Check if questions have required columns
            question_columns = pd.Series(['text', 'answer_id'])
            if not question_columns.isin(questions.columns).all():
                raise ValueError('Questions should have columns: text, answer_id')
            
            # Check if answers have required columns
            answer_columns = pd.Series(['id', 'text'])
            if not answer_columns.isin(answers.columns).all():
                raise ValueError('Answers should have columns: id, text')
            
            # Embed questions
            questions['embedding'] = questions['text'].apply(self.calculate_sentence_embedding)
        elif not questions and not answers:
            # Get questions and answers from database
            questions = pd.DataFrame(Question.search({})).drop(columns=['_id'])
            answers = pd.DataFrame(Answer.search({})).drop(columns=['_id'])
            
            # Embed questions if not already embedded
            questions['embedding'] = questions.apply(self.calculate_sentence_embedding_db, axis=1)
        else:
            raise ValueError('Either both questions and answers should be provided or none of them')
        
        # Set questions and answers as attributes
        self.questions = questions[['text', 'embedding', 'answer_id']]
        self.answers = answers[['id', 'text']]

    def predict(self, input_question: str, nth_similar: int = 1) -> str:
        '''
        @param input_question: The question to answer
        @param nth_similar: The nth most similar question to return the answer to
        @return: The answer to the question
        '''
        # Embed input question
        input_embedding = self.calculate_sentence_embedding(input_question)

        # Calculate cosine similarity of question embeddings
        questions = self.questions.copy()
        questions['similarities'] = questions['embedding'].apply(lambda x: self.calculate_cosine_similarity(input_embedding, x))
        
        # Find the most (/nth) similar question
        questions = questions.loc[questions['similarities'] > 0.55] 
        questions = questions.sort_values(by='similarities', ascending=False)
        questions = questions.drop_duplicates(subset=['answer_id'], keep='first') 
        questions = questions.reset_index(drop=True)
        if nth_similar > questions.shape[0]:
            return None, None, None
        nth_similar_question = questions.iloc[nth_similar-1]
        
        # Get answer to the most (/nth) similar question
        answer_id = nth_similar_question['answer_id']
        answer = self.answers.loc[self.answers['id'] == answer_id]['text'].values[0]
        nth_similar_question_text = nth_similar_question['text']

        return answer, answer_id, nth_similar_question_text


if __name__ == '__main__':
    legalpha = LegalphaSemSearch()
    print(legalpha.predict('What is the procedure for terminating a rental contract?'))
