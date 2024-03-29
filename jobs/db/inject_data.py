import pandas as pd
from pymongo import MongoClient
import os
from models import Answer, UserQuestion
from utils.data import read_csv


# Set models to inject data for
MODELS = [Answer, UserQuestion]

def inject_data():
    # Connect to MongoDB
    client = MongoClient(os.environ.get('MONGODB_URI'))
    db = client.get_database(os.environ.get('MONGO_INITDB_ROOT_DATABASE'))

    # Create required collections if they don't exist
    collection_names = db.list_collection_names()
    for model in MODELS:
        collection = model.__collection__
        if collection not in collection_names:
            db.create_collection(collection)

            if not os.path.exists(f'./data/{collection}.csv'):
                continue

            data = read_csv(f'./data/{collection}.csv').to_dict(orient='records')

            # Create records
            for record in data:
                model(**record).create()


if __name__ == '__main__':
    inject_data()
