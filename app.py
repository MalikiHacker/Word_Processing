# coding=utf8
from flask import Flask,render_template,url_for,request
import json
import re
import numpy as np
from keras_preprocessing.text import tokenizer_from_json
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

"""
#if some truble with tensorflow-gpu try that
physical_devices = tf.config.experimental.list_physical_devices('GPU') 
for physical_device in physical_devices: 
    tf.config.experimental.set_memory_growth(physical_device, True)
"""


# Import CNN Model
model = load_model('model_2/word_processing.h5')
maxlen = 300
id_to_category = {0:'business', 1:'entertainment', 2:'politics', 3:'sport', 4:'tech'}

with open('model_2/tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)
    
def preprocess_text(sen):
    # Lowercase
    sentence = sen.lower()
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Zäöüß]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

application = app = Flask(__name__)

@application.route('/')
def home():
	return render_template('home.html')

@application.route('/predict',methods=['POST'])

def predict():

    id_to_category = {0:'business', 1:'entertainment', 2:'politics', 3:'sport', 4:'tech'}
    
    if request.method == 'POST':
        input_message = request.form['message']
        message = preprocess_text(str(input_message))
        if message.strip() == "":
            result="Please Enter a Headline"
        if len(message.split()) < 4:
            result=["Please Enter more words"]
        else:
            my_input = [message]
            input_sequences = tokenizer.texts_to_sequences(my_input)
            input_pad = pad_sequences(input_sequences, padding='post', maxlen=maxlen)
            preds = model.predict(input_pad)[0]
            
            pred_classes = np.argsort(preds)[-5:][::-1]

            classes = [id_to_category[i] for i in pred_classes]
            props   = preds[pred_classes]

            result={}
            for c, p in zip(classes, props):
                #result.append("{} {:.2f} %".format(c,p*100))
                result[c] = round(p*100,2)
                
    return render_template('result.html', mess = input_message, classes = list(result.keys()), props=list(result.values()) )


if __name__ == '__main__':
    #app.run(host="0.0.0.0",port=5000,debug=True)
    app.run(port=5000,debug=True)