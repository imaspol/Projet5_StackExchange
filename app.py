#importing the necessary libraries for deployment
from flask import Flask, request, jsonify, render_template
import joblib

# Pre-processing
from collections import Counter, defaultdict

import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag

import matplotlib.pylab as plt
import string

import re
import pandas as pd

stop_words = set(stopwords.words('english'))
for i in ["'m", "'ve", "'s", "'d", "'re", "'ll", "'t"]: 
    stop_words.add(i)

def preprocessing(extracted_text):
    '''The preprocessing step include:
    tokenisation
    deletion of punctuation
    deletion of stop-words'''
    # Tokenisation
    text_tokenized = nltk.word_tokenize(extracted_text)
    #print('text_tokenized : ', text_tokenized)
    # Punctuation deletion
    import string
    text_stripped = []
    for _word in text_tokenized:
        text_stripped.append(''.join(_c for _c in _word if _c not in string.punctuation))
        #if _word not in string.punctuation:
            # word to small letters
        #    text_stripped.append(_word)
    #print('text_stripped : ', text_stripped)

    filtered_sentence = []
    
    # Stop-words deletion
    for _word in text_stripped:
        if _word and _word.lower() not in stop_words and not _word.isdigit() :
            filtered_sentence.append(_word)
#     print(filtered_sentence)
    return filtered_sentence

#naming our app as app
app= Flask(__name__)

#loading the pickle file for creating the web app
model= joblib.load(open("finalized_model.sav", "rb"))

#defining the different pages of html and specifying the features required to be filled in the html form
@app.route("/")
def home():
    return render_template("index.html")

#creating a function for the prediction model by specifying the parameters and feeding it to the ML model
@app.route("/predict", methods=["POST"])
def predict():
    #specifying our parameters as data type float
    # convertir en dataframe
    ans=request.form
    final_features= ans['title'] + ans['body']
    serie_usertext = pd.Series(data=final_features)
    prediction= model.predict(serie_usertext)
    output= prediction
    return render_template("index.html", prediction_text= "flower is {}".format(output))
#running the flask app
if __name__ == "__main__":
    app.run(debug=True)