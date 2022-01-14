#importing the necessary libraries for deployment
from flask import Flask, request, jsonify, render_template
import joblib
import sys

# Pre-processing
from collections import Counter, defaultdict

import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag

import matplotlib.pylab as plt
import string

import re
import datetime
import logging
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()

stop_words = set(stopwords.words('english'))
for i in ["'m", "'ve", "'s", "'d", "'re", "'ll", "'t"]: 
    stop_words.add(i)

logging.getLogger().setLevel(logging.INFO)

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
app = Flask(__name__)

# HACK FOR PYTHON ANYWHERE
if not hasattr(sys.modules['__main__'], 'preprocessing'):
    setattr(sys.modules['__main__'], 'preprocessing', preprocessing)

#loading the pickle file for creating the web app
logging.info('load model')
model= joblib.load(open("finalized_model.sav", "rb"))

# fitted_binarizer
logging.info('load binarizer')
fitted_mlb = joblib.load(open("fitted_binarizer.sav", "rb"))

logging.info('finished loading')
#defining the different pages of html and specifying the features required to be filled in the html form
@app.route("/")
def home():
    _default_title = "What programming language to choose?"
    _default_body = """<p>Hi! What programming language to choose between Python, Java and C++?</p>"""
    return render_template("index.html", title_text=_default_title, body_text=_default_body)

#creating a function for the prediction model by specifying the parameters and feeding it to the ML model
@app.route("/predict", methods=["POST"])
def predict():
    #specifying our parameters as data type float
    # convertir en dataframe
    ans=request.form
    _title = ans['title'].strip()
    _body = ans['body'].strip()
    final_features = ans['title'] + ans['body']
    serie_usertext = pd.Series(data=final_features)
    prediction = model.predict(serie_usertext)
    #print(prediction)
    output = fitted_mlb.inverse_transform(prediction)
    print(output)
    #print(', '.join(output[0]))
    date = datetime.datetime.now()
    #logging.info(prediction)
    return render_template("index.html",
        title_text=_title, body_text=_body,
        prediction_text= f"{str(date)}: Tags are {(', '.join(output[0]))}"
    )
 

#running the flask app
if __name__ == "__main__":
    logging.info('Running Flask Server')
    app.run(debug=True)