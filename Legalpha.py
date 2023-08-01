import pandas as pd
from simpletransformers.language_representation import RepresentationModel
from sklearn.metrics.pairwise import cosine_similarity
from models import Question, Answer

class Legalpha:
    bert = RepresentationModel('bert', 'bert-base-uncased', use_cuda=False)

    def calculate_sentence_embedding(self, sentence: str) -> list:
        '''
        @param sentence: The sentence to embed
        @return: The embedding of the sentence

        This function takes in a sentence and returns the embedding of the sentence.
        '''

        return Legalpha.bert.encode_sentences([sentence], combine_strategy='mean').tolist()
    
    def calculate_cosine_similarity(self, embedding1: str, embedding2: str) -> float:
        '''
        @param embedding1: The first embedding
        @param embedding2: The second embedding
        @return: The cosine similarity of the two embeddings

        This function takes in two embeddings and returns the cosine similarity of the two embeddings.
        '''

        return cosine_similarity(embedding1, embedding2)[0][0]

    def answer(self, input_question: str, nth_similar: int = 1) -> str:
        '''
        @param input_question: The question to answer
        @return: The answer to the question

        This function takes in a question and returns the answer to the question.
        1. The question is embedded using BERT.
        2. The most similar question in database is found by comparing embeddings with cosine similarity.
        3. The answer to the most similar question is returned.
        '''

        # Embed input question
        input_embedding = self.calculate_sentence_embedding(input_question)

        questions = Question.search({'answer_id': {'$exists': True}})
        questions_processed = pd.DataFrame(columns=['question', 'answer_id', 'similarity'])
        # Iterate over each question in the database 
        for question in questions:
            # Skip question if no answer is available
            if pd.isna(question.get('answer_id')):
                continue

            # Embed question if not already embedded
            question_embedding = question.get('embedding')
            if not question_embedding:
                question_embedding = self.calculate_sentence_embedding(question.get('text'))
                question['embedding'] = question_embedding
                question.pop('_id')
                Question(**question).update()

            # Calculate cosine similarity of question embeddings
            similarity = self.calculate_cosine_similarity(input_embedding, question_embedding)
            questions_processed = pd.concat([
                questions_processed, 
                pd.DataFrame({'question': [question.get('text')], 
                              'answer_id': [question.get('answer_id')], 
                              'similarity': [similarity]})
            ])

        # Get answer to most (/nth) similar question
        questions_processed = questions_processed.loc[questions_processed.similarity > 0.55]
        questions_processed = questions_processed.sort_values(by='similarity', ascending=False)
        questions_processed = questions_processed.drop_duplicates(subset=['answer_id'], keep='first')
        questions_processed = questions_processed.reset_index(drop=True)

        if questions_processed.shape[0] < nth_similar:
            return None, None

        row_of_nth_similar = questions_processed.loc[nth_similar-1]
        nth_similar_question = row_of_nth_similar['question']
        answer_id = row_of_nth_similar['answer_id']
        answer = Answer.search({'id': answer_id}).next().get('text')

        print('Input question: ', input_question)
        print('Row of nth similar:', row_of_nth_similar)
        print('Nth similar question: ', nth_similar_question)
        print('Answer: ', answer)
        print('Similarity: ', row_of_nth_similar['similarity'])

        return answer, nth_similar_question


if __name__ == '__main__':
    legalpha = Legalpha()
    print(legalpha.answer('What is the procedure for terminating a rental contract?'))
