from models import UserQuestion, Answer

def strip_answer_from_pointers(query_answer: str, all_answer_pointers: list):
    for answer_pointer in all_answer_pointers:
        query_answer_split = query_answer.split(answer_pointer)
        if len(query_answer_split) > 1:
            query_answer = query_answer_split[1] # Get the answer after the answer pointer
            query_answer = query_answer[1:] # Remove the space after the answer pointer
            query_answer = query_answer[0].upper() + query_answer[1:] # Capitalize the first letter
            break

    return query_answer

def get_id_from_query_answer(query_answer: str, all_answer_pointers: list):
    query_answer = strip_answer_from_pointers(query_answer, all_answer_pointers)
    answer_id = Answer.search({'text': query_answer})[0]['id']
    return answer_id

def create_user_question(query_question):
    user_question = UserQuestion(text=query_question)
    user_question.create()
    user_question.read()
    return user_question
    
def handle_feedback(query_question, answer_id, feedback):
    user_questions = UserQuestion.search({'text': query_question})
    updated = False

    for user_question in user_questions:
        print('User Question 1: ')
        print(user_question)
        user_question.pop('_id')
        print('User Question 2: ')
        print(user_question)
        user_question = UserQuestion(**user_question)
        user_question.first_answer_id = user_question.first_answer_id if user_question.first_answer_id is not None else answer_id
        user_question.first_feedback = user_question.first_feedback if user_question.first_feedback is not None else feedback
        user_question.last_answer_id = answer_id
        user_question.last_feedback = feedback
        user_question.n_answers = user_question.n_answers + 1
        user_question.update()
        updated = True

    return updated