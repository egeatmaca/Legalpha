import numpy as np
from simpletransformers.language_representation import RepresentationModel
from simpletransformers.question_answering import QuestionAnsweringModel
from sklearn.metrics.pairwise import cosine_similarity
from models import Question, Answer

class Legalpha:
    representation_model = RepresentationModel('bert', 'bert-base-uncased', use_cuda=False)
    qa_model = QuestionAnsweringModel('bert', 'bert-base-uncased', use_cuda=False)

    def answer(self, input_question: str) -> str:
        '''
        @param input_question: The question to answer
        @return: The answer to the question

        This function takes in a question and returns the answer to the question.
        1. The question is embedded using BERT.
        2. The most similar question in database is found by comparing embeddings with cosine similarity.
        3. The answer to the most similar question is returned.
        '''

        # Embed input question
        input_embedding = Legalpha.representation_model.encode_sentences([input_question], combine_strategy='mean').tolist()

        print(f'Input question: {input_question}')

        questions = Question.search({'answer_id': {'$exists': True}})
        max_similarity = -1
        most_similar_question = None
        # Iterate over each question in the database 
        for question in questions:
            # Skip question if no answer is available
            if np.isnan(question.get('answer_id')):
                continue

            # Embed question if not already embedded
            question_embedding = question.get('embedding')
            if not question_embedding:
                question_embedding = Legalpha.representation_model.encode_sentences([question.get('text')], combine_strategy='mean').tolist()
                question['embedding'] = question_embedding
                question.pop('_id')
                Question(**question).update()

            # Calculate cosine similarity of question embeddings
            cos_similarity = cosine_similarity(input_embedding, question_embedding)
            similarity = cos_similarity[0][0]

            print(f'Question: {question.get("text")}')
            print(f'Similarity: {similarity}')

            # Update the most similar question and max similarity if necessary
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_question = question

                print(f'New max similarity: {max_similarity}')
                print(f'New most similar question: {most_similar_question.get("text")}')

        print(f'Most similar question: {most_similar_question.get("text")}')
        print(f'Answer ID: {most_similar_question.get("answer_id")}')
        print(f'Type of Answer ID: {type(most_similar_question.get("answer_id"))}')

        full_answer = Answer.search({'id': most_similar_question['answer_id']}).next().get('text')
        specific_answer = Legalpha.qa_model.predict([{'context': full_answer, 'qas': [{'question': input_question, 'id': 0}]}])[0]

        return specific_answer
    
if __name__ == '__main__':
    legalpha = Legalpha()
    print(legalpha.answer('What is the procedure for terminating a rental contract?'))
