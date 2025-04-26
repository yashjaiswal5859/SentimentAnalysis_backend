import os
import requests
import gdown
def download_from_drive(file_id, destination):
    if os.path.exists(destination):
        print(f"{destination} already exists, skipping download.")
        return
    print(f"Downloading {destination}...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False)

    # Check file size after download
    if os.path.exists(destination):
        size_in_bytes = os.path.getsize(destination)
        size_in_mb = size_in_bytes / (1024 * 1024)  # convert to MB
        if size_in_bytes == 0:
            print(f"Error: {destination} was downloaded but is empty! ❌")
        else:
            print(f"Downloaded {destination} ✅")
            print(f"File size: {size_in_mb:.2f} MB")
    else:
        print(f"Error: {destination} was not downloaded at all! ❌")
    return
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)
    
    print(f"Downloaded {destination}")
    size_in_bytes = os.path.getsize(destination)
    size_in_mb = size_in_bytes / (1024 * 1024)  # convert to MB
    print(f"File size: {size_in_mb:.2f} MB")

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None
def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
download_from_drive('1SyckEYNFmscalIIEtiuw4H6gWgDFUn6B', 'Bidirection.h5')
download_from_drive('1zaf264WY_WNRcijoXDBnIRFtTBT62J47', 'LSTM.h5')
download_from_drive('1Agk7OWPY3aIrEbHwfpD-rSm1COMiEoK1', 'tokenizer.pkl')


from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from collections import defaultdict
import nltk
import pandas as pd 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import re
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes

# Load your Keras model
bidir = tf.keras.models.load_model('Bidirection.h5')  # your .h5 model
lstm_model = tf.keras.models.load_model('LSTM.h5')  # your .h5 model

def readText():
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer
    

word_index=readText()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.remove('not')
def remove_special(text):
    return re.sub('\[[^&@#!]]*\]','',text)
def removeurl(text):
    return re.sub(r'http\S+','',text)
def contractionExp(text):
    text = re.sub(r"won't", "would not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"ain't", "is not", text)
    text = re.sub(r"(\w+)'ll", r"\1 will", text)
    text = re.sub(r"(\w+)'ve", r"\1 have", text)
    text = re.sub(r"(\w+)'re", r"\1 are", text)
    text = re.sub(r"(\w+)'d", r"\1 would", text)
    text = re.sub(r"(\w+)'s", r"\1 is", text)  # note: might catch possessives too
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"’t", " not", text)  # handles curly apostrophe
    text = re.sub(r"’re", " are", text)
    text = re.sub(r"’s", " is", text)
    text = re.sub(r"’ll", " will", text)
    return text
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text=remove_special(text)
    text=removeurl(text)
    text=contractionExp(text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return tokens
MAX_LEN = 164  # max

def texttoSeq(text):
    X_= preprocess(text)

    sequence = word_index.texts_to_sequences(X_)
    padded_sequences = np.zeros((len(sequence), MAX_LEN), dtype=int)
    for i in range(len(sequence)):
        padded_sequences[i][0:len(sequence[i])]=sequence[i]
    return padded_sequences

@app.route('/lstm', methods=['POST'])
def lstm():
    data = request.get_json()
    text = data['text']
    seq=texttoSeq(text)
    pred=lstm_model.predict(seq)
    return jsonify({'prediction': float(pred[0])})

@app.route('/bilstm', methods=['POST'])
def bilstm():
    data = request.get_json()
    text = data['text']
    seq=texttoSeq(text)
    pred=bidir.predict(seq)
    print(text,seq)
    return jsonify({'prediction': float(pred[0])})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
