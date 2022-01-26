# Importing Libraries

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import motor.motor_asyncio
# from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')
# from nltk.stem import LancasterStemmer
# from sklearn.feature_extraction.text import CountVectorizer
# import pandas as pd
# import nltk
# nltk.download('punkt')
# import re
# from nltk.corpus import stopwords
# nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn import model_selection, preprocessing
# from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import pickle
from bson import json_util
import json
import datetime

# Loading all the models/vectorizer/encodings

loadModel = pickle.load(open("XGBoostClassifier.pickle.dat", "rb"))
vectorizer = pickle.load(open("tfidfVectorizer.pickle.dat", "rb"))
encoding = pickle.load(open("Encodings.pickle.dat", "rb"))


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def getDataFromDB(colName):
    CONNECTION_STRING = "mongodb+srv://OvaizAli:123@cronyai.idwl9.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"

    client = motor.motor_asyncio.AsyncIOMotorClient(CONNECTION_STRING, serverSelectionTimeoutMS=5000)

    database = client.CronyAI
    collection = database.get_collection(colName)
    cursor = collection.find()
    
    try:
        data = await cursor.to_list(None)
        return json.loads(json_util.dumps(data))

    except Exception:
        return "Unable to connect to the server."


async def addDataFromDB(colName, dataObj):  
    CONNECTION_STRING = "mongodb+srv://OvaizAli:123@cronyai.idwl9.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"

    client = motor.motor_asyncio.AsyncIOMotorClient(CONNECTION_STRING, serverSelectionTimeoutMS=5000)
    database = client.CronyAI
    collection = database.get_collection("action")
   
    try:
        result = await collection.insert_one(dataObj)

        return "Success"

    except Exception:
        return "Unable to connect to the server."


@app.get("/")
def info():
    return "Hello Buddy! Use /{userInput} to get the results from the API. e.g https://cronyfastapi.herokuapp.com/Who is at the store today?"

@app.get("/query/{userInput}")
def cmdQuery(userInput : str):
    userInput = userInput.lower()
    vect = vectorizer.transform([userInput])
    # print(max(vect.toarray()[0]))
    if max(vect.toarray()[0]) >= 0.5:
        outQuery = loadModel.predict(vectorizer.transform([userInput]))
        # print(encoding.inverse_transform(outQuery))
    
        return {"userInput": userInput,
        "Query" : encoding.inverse_transform(outQuery)[0], 
        "conThresh" : round(max(vect.toarray()[0]), 2)}
    else:
        
        return {"Sorry I didn't get you!"}

@app.get("/addActions/")
# def addActions(actionName: str, actionType: str, dataAdded: int):
def addActions(actionName: str, actionType: str):
    action = {
        "actionName" : actionName,
        "actionType" : actionType,
        "dateAdded" : datetime.datetime.utcnow()
    }

    print(action)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(addDataFromDB("action", action))


@app.get("/getNotFound")
def getNotFound():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(getDataFromDB("not_found"))


@app.get("/getActions")
def getActions():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(getDataFromDB("action"))

