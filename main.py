from flask import Flask
from flask import request
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

with open('COVID.json') as file:
    data = json.load(file)
from sklearn.preprocessing import LabelEncoder

training_sentences = []
training_labels = []
labels = []
responses = []


for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

enc = LabelEncoder()
enc.fit(training_labels)
training_labels = enc.transform(training_labels)
vocab_size = 10000
embedding_dim = 16
max_len = 20
trunc_type = 'post'
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token) # adding out of vocabulary token
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, truncating=trunc_type, maxlen=max_len)
classes = len(labels)

app = Flask(__name__)
model = tf.keras.models.load_model(
    "z_bot", custom_objects=None, compile=True, options=None
)

@app.route("/")
def index():
    covidResponse = request.args.get("covidResponse", "")
    if covidResponse:
        answer = generate_answer(covidResponse)
    else:
        answer = ""
    return (
        """
        <style>
            .mainTitle{
                    color: black;
                    font-family: Oswald, sans-serif !important;
                    font-size: 3rem;
                    text-transform: uppercase;
                    font-weight: 700;
                    margin: 10px 50px 10px 50px;
            }
            
            .formPrompt{
                    color: #232b2b;
                    margin: 25px 50px 10px 50px;
                    font-size: 1.5rem;
                    font-weight: 700;
                    text-transform: uppercase;
                    font-family: Oswald, sans-serif !important;
            }
            input[type=text] {
                    width: 100%;
                    padding: 12px 20px;
                    margin: 8px 0;
                    box-sizing: border-box;
                    font-family: Oswald, sans-serif !important;
                    font-size: 1rem;
        }   
        
            .inputText{
                    font-family: Oswald, sans-serif !important;
                    font-size: 1rem;
                    font-weight: 500;   
                    color: #232b2b;
                    margin: 25px 50px 10px 50px;
        }
        
            .answerText{
                    color: #232b2b;
                    margin: 50px 50px 10px 50px;
                    font-size: 1.5rem;
                    font-weight: 700;
                    text-transform: uppercase;
                    font-family: Oswald, sans-serif !important;        
        }  
            .submitButton{
                    text-transform: uppercase;
                    display: inline-block;
                    padding: 0.35em 1.2em;
                    border: 0.1em solid black;
                    margin: 0 0.3em 0.3em 0;
                    border-radius: 0.12em;
                    box-sizing: border-box;
                    text-decoration: none;
                    font-family: 'Roboto',sans-serif;
                    font-weight: 300;
                    color: white;
                    text-align: center;
                    font-size: 1.5rem;
                    background-color: black;    
                    transition: all 0.3s ease-in-out;    
        } 

            .submitButton:hover{
                    cursor: pointer;  
                    transform: scale(1.05);                  
        }
        
        
        </style>
        <head>
         <title>Ontario COVID Chatbot</title>
        </head>
        <br>
        <h1 class="mainTitle" >Ontario's COVID Help AI Chatbot</h1>
        <hr STYLE="background-color:#000000; height:5px; width:95%;">
        <form action="" method="get" class="formPrompt" autocomplete="off">
            Ask Question about Covid Protocols, Restrictions, and other General COVID Information:
             <br>
             <input type="text" name="covidResponse" class ="inputText">
            <input type="submit" value="Ask Question" class ="submitButton">
        </form>    
        <h1 class="answerText">Answer: </h1>
            """
        + '''<h1 style = "color: black; margin: 25px 50px 10px 50px; font-size: 1.5rem; font-weight: 700; font-family: Oswald, sans-serif !important;">''' + answer + '''</h1>'''
    )

def generate_answer(covidResponse):
    try:
        result = model.predict(pad_sequences(tokenizer.texts_to_sequences([covidResponse]),
                                             truncating=trunc_type, maxlen=max_len))
        category = enc.inverse_transform([np.argmax(result)])  # labels[np.argmax(result)]
        for i in data['intents']:
            if i['tag'] == category:
                answer = np.random.choice(i['responses'])
                return answer
    except ValueError:
        return "invalid input"

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)