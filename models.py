from pymongo import MongoClient
import os
from abc import ABC, abstractmethod


# Connect to MongoDB
client = MongoClient(os.environ.get('MONGODB_URI'))
db = client.get_database(os.environ.get('MONGO_INITDB_ROOT_DATABASE'))


class Model(ABC):
    __collection__ = None

    @abstractmethod
    def __init__(self, id: int = None):
        if id is None:
            max_id_records = type(self).search({'id': {'$exists': True}}).sort('id', -1).limit(1)

            id = 0
            for record in max_id_records:
                id = record.get('id') + 1

        self.id = id

    def to_dict(self):
        return self.__dict__

    def create(self):
        if not self.__collection__:
            raise NotImplementedError('Cannot save model without collection name')
        db[self.__collection__].insert_one(self.to_dict())

    def read(self):
        if not self.__collection__:
            raise NotImplementedError('Cannot read model without collection name')
        return db[self.__collection__].find_one({'id': self.id})
    
    def update(self):
        if not self.__collection__:
            raise NotImplementedError('Cannot update model without collection name')
        db[self.__collection__].update_one({'id': self.id}, {'$set': self.to_dict()})

    def delete(self):
        if not self.__collection__:
            raise NotImplementedError('Cannot delete model without collection name')
        db[self.__collection__].delete_one({'id': self.id})

    @classmethod
    def search(cls, query):
        collection = cls.__collection__
        if not collection:
            raise NotImplementedError('Cannot search model without collection name')
        return db[collection].find(filter=query)
    

class Answer(Model):
    __collection__ = 'answers'

    def __init__(self,  id: int, text: str = None, topic: str = None):
        self.text = text
        self.topic = topic
        super().__init__(id)


class Question(Model):
    __collection__ = 'questions'
    
    def __init__(self, text: str, id: int = None, embedding: list = None, answer_id: str = None):
        self.text = text
        self.embedding = embedding
        self.answer_id = answer_id
        super().__init__(id)

    def get_answer(self):
        return Answer(id= self.answer_id)
    

class UserQuestion(Model):
    __collection__ = 'user_questions'

    def __init__(self, id: int = None, 
                 text: str=None, last_answer: int = None,
                 first_answer_received_feedback: int = None, first_feedback: bool = None, 
                 last_answer_received_feedback: int = None, last_feedback: bool = None, 
                 n_answers: int = 0):
        self.text = text
        self.last_answer = last_answer
        self.first_answer_received_feedback = first_answer_received_feedback
        self.first_feedback = first_feedback
        self.last_answer_received_feedback = last_answer_received_feedback
        self.last_feedback = last_feedback
        self.n_answers = n_answers
        super().__init__(id)