# Importing Libraries

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import motor.motor_asyncio
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import LancasterStemmer
import re
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, preprocessing
from sklearn.metrics import accuracy_score
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
        return "Unable To Connect To The Server"



async def getPickleFilesFromDB(modelType, file):
    CONNECTION_STRING = "mongodb+srv://OvaizAli:123@cronyai.idwl9.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"

    client = motor.motor_asyncio.AsyncIOMotorClient(CONNECTION_STRING, serverSelectionTimeoutMS=5000)

    database = client.CronyAI
    collection = database.get_collection("pickle_files")
    # cursor = collection..find_one({'modelType': {'$eq': modelType}})
    
    try:
        data = await collection.find_one({'modelType': {'$eq': modelType}})
        return data[file]

    except Exception:
        return "Unable To Connect To The Server"



async def addDataToDB(colName, dataObj):  
    CONNECTION_STRING = "mongodb+srv://OvaizAli:123@cronyai.idwl9.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"

    client = motor.motor_asyncio.AsyncIOMotorClient(CONNECTION_STRING, serverSelectionTimeoutMS=5000)
    database = client.CronyAI
    collection = database.get_collection(colName)
   
    try:
        result = await collection.insert_one(dataObj)
        return "Successfully Added Your Data"

    except Exception:
        return "Unable To Connect To The Server"



async def delDataFromDB(colName, userInput):  
    CONNECTION_STRING = "mongodb+srv://OvaizAli:123@cronyai.idwl9.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"

    client = motor.motor_asyncio.AsyncIOMotorClient(CONNECTION_STRING, serverSelectionTimeoutMS=5000)
    database = client.CronyAI
    collection = database.get_collection(colName)

    try:
        cursor = collection.delete_many({'phraseInput': {'$eq': userInput}})
        return "Successfully Deleted Your Data"

    except Exception:
        return "Unable To Connect To The Server"



async def delPickleFiles(modelType):  
    CONNECTION_STRING = "mongodb+srv://OvaizAli:123@cronyai.idwl9.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"

    client = motor.motor_asyncio.AsyncIOMotorClient(CONNECTION_STRING, serverSelectionTimeoutMS=5000)
    database = client.CronyAI
    collection = database.get_collection("pickle_files")

    try:
       cursor = collection.delete_many({'modelType': {'$eq': modelType}})
       return "Successfully Deleted Your Files"

    except Exception:
        return "Unable To Connect To The Server"



def modelTrain(actionType):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        df_data = pd.DataFrame(loop.run_until_complete(getDataFromDB('mapped_action')))
        df_data = df_data[df_data['actionType'] == actionType][['phraseInput', 'actionName']]

        data = df_data['phraseInput'].copy()

        stop_words = stopwords.words('english')

        lemmatizer = WordNetLemmatizer()
        index = 0
        for row in data:
            print(row)
            filtered_sentence = []
            row = row.lower()
            sentence = re.sub(r'[^\w\s]', '', row)
            words = nltk.word_tokenize(sentence)
            words = [w for w in words if not w in stop_words]
            for word in words:
                # print(word)
                filtered_sentence.append(lemmatizer.lemmatize(word))
            data.iloc[index] = ','.join(filtered_sentence).replace(",", " ")
            index += 1
        
        print(data)
        df_data['phraseInput'] = data

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(df_data['phraseInput'])

        # pickle.dump(vectorizer, open(actionType + "tfidfVectorizer.pickle.dat", "wb"))
        pickled_vectorizer = pickle.dumps(vectorizer)

        print("Vectorizer")

        train_x, valid_x, train_y, valid_y = model_selection.train_test_split(vectors, df_data['actionName'], random_state = 42, test_size = 0.20, stratify = df_data['actionName'])
        encoder = preprocessing.LabelEncoder()
        train_y = encoder.fit_transform(train_y)
        valid_y = encoder.transform(valid_y)
        
        # pickled_encoder = pickle.dumps(encoder, open(actionType + "Encodings.pickle.dat", "wb"))
        pickled_encoder = pickle.dumps(encoder)

        print("Encoder")

        xgb = XGBClassifier(use_label_encoder=False,learning_rate=0.4,max_depth=7)
        xgb.fit(train_x, train_y)

        # pickle.dump(xgb, open(actionType + "XGBoostClassifier.pickle.dat", "wb"))
        pickled_model = pickle.dumps(xgb)

        print("Model")

        pickle_files = {
            "model" : pickled_model,
            "vectorizer" : pickled_vectorizer,
            "encodings" : pickled_encoder,
            "modelType" : actionType,
            "dateAdded" : datetime.datetime.utcnow()
        }

        print(pickle_files)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        print("Here")

        if loop.run_until_complete(delPickleFiles(actionType)) == "Successfully Deleted Your Files":
            loop.run_until_complete(addDataToDB("pickle_files", pickle_files))
            return "Successfully Trained and Updated The Model"
        else:
            loop.run_until_complete(addDataToDB("pickle_files", pickle_files))
            return "Successfully Trained The Model"

    except Exception as e: 
        return e
    # except:
    #     return "Unable To Connect To The Server"

############################################################## API FUNCTIONS ###############################################################

@app.get("/")
def info():
    return "Hello Buddy! Welcome to https://cronyfastapi.herokuapp.com/"



@app.get("/query/")
def cmdQuery(userInput : str, modelType : str):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loadModel = pickle.loads(loop.run_until_complete(getPickleFilesFromDB(modelType, "model")))
        vectorizer = pickle.loads(loop.run_until_complete(getPickleFilesFromDB(modelType, "vectorizer")))
        encoding = pickle.loads(loop.run_until_complete(getPickleFilesFromDB(modelType, "encodings")))
          
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

    except:
        return "Sorry, We donot have Model Trained For This Model Type"



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



# @app.get("/addMapAction/")
# def addMapAction(phraseInput : str, actionName: str, actionType: str, ):
#     mapAction = {
#         "phraseInput" : phraseInput,
#         "actionName" : actionName,
#         "actionType" : actionType,
#         "dateAdded" : datetime.datetime.utcnow()
#     }

#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     try:
#         loop.run_until_complete(addDataToDB("mapped_action", mapAction))
#         return "Successfully Added into Mapped_Action"

#     except:
#         return "Unable To Connect To The Server"



@app.get("/mapAction/")
def mapAction(phraseInput : str, actionName: str, actionType: str):
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
            return "Unable To Connect To The Server"



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



@app.get("/modelTrain")
def getModelTrained(actionType: str):
    return modelTrain(actionType)




