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

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

######################################################## UTILITY FUNCTIONS ##############################################

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
        return "Unable To Connect To The Server."



async def addDataToDB(colName, dataObj):  
    CONNECTION_STRING = "mongodb+srv://OvaizAli:123@cronyai.idwl9.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"

    client = motor.motor_asyncio.AsyncIOMotorClient(CONNECTION_STRING, serverSelectionTimeoutMS=5000)
    database = client.CronyAI
    collection = database.get_collection(colName)
   
    try:
        result = await collection.insert_one(dataObj)
        return "Successfully Added Your Data"

    except Exception:
        return "Unable To Connect To The Server."



async def delDataFromDB(colName, userInput):  
    CONNECTION_STRING = "mongodb+srv://OvaizAli:123@cronyai.idwl9.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"

    client = motor.motor_asyncio.AsyncIOMotorClient(CONNECTION_STRING, serverSelectionTimeoutMS=5000)
    database = client.CronyAI
    collection = database.get_collection(colName)

    try:
        cursor = collection.delete_many({'phraseInput': {'$eq': userInput}})
        return "Successfully Deleted Your Data"

    except Exception:
        return "Unable To Connect To The Server."

############################################################## API FUNCTIONS ###############################################################

@app.get("/")
def info():
    return "Hello Buddy! Welcome to https://cronyfastapi.herokuapp.com/"



@app.get("/query/")
def cmdQuery(userInput : str, modelType : str):
    if modelType == "Admin":
        loadModel = pickle.load(open("AdminXGBoostClassifier.pickle.dat", "rb"))
        vectorizer = pickle.load(open("AdmintfidfVectorizer.pickle.dat", "rb"))
        encoding = pickle.load(open("AdminEncodings.pickle.dat", "rb"))

        userInput = userInput.lower()
        vect = vectorizer.transform([userInput])
        if max(vect.toarray()[0]) >= 0.5:
            outQuery = loadModel.predict(vectorizer.transform([userInput]))
            return {"userInput": userInput,
                "Query" : encoding.inverse_transform(outQuery)[0], 
                "conThresh" : round(max(vect.toarray()[0]), 2)}

        else:
            userInput = userInput.upper()

            notFound = {
                    "phraseInput" : userInput,
                    "dateAdded" : datetime.datetime.utcnow()
                    }
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(addDataToDB("not_found", notFound))
            return "Sorry I didn't get you!"

    else:
        return "Sorry, we donot have model trained for you"



@app.get("/addActions/")
def addActions(actionName: str, actionType: str):
    action = {
        "actionName" : actionName,
        "actionType" : actionType,
        "dateAdded" : datetime.datetime.utcnow()
    }

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(addDataToDB("action", action))



@app.get("/mapAction/")
def mapAction(phraseInput : str, actionName: str, actionType: str, ):
    mapAction = {
        "phraseInput" : phraseInput,
        "actionName" : actionName,
        "actionType" : actionType,
        "dateAdded" : datetime.datetime.utcnow()
    }

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    if loop.run_until_complete(delDataFromDB("not_found", phraseInput)) == "Successfully Deleted Your Data":
        try:
            loop.run_until_complete(addDataToDB("mapped_action", mapAction))
            return "Successfully Mapped Your Action"
        except:
            return "Unable To Connect To The Server."



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

