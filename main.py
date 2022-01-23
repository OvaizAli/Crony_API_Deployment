import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

@app.get("/")
def info():
    return "Hello Buddy! Use /{userInput} to get the results from the API. e.g https://cronyfastapi.herokuapp.com/Who is at the store today?"

@app.get("/{userInput}")
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
    