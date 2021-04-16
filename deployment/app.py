import flask
import json
import tensorflow as tf
import numpy as np
import random
import spacy
import pickle
nlp = spacy.load('en_core_web_sm')

model = tf.keras.models.load_model('chatbot.h5')
with open('tokenizer.json', 'rb') as f:
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())
with open('intent_doc.pickle', 'rb') as f:
    intent_doc = pickle.load(f)
with open('trg_index_word.pickle', 'rb') as f:
    trg_index_word = pickle.load(f)

def response(sentence):
    sent_seq = []
    doc = nlp(repr(sentence))
    
    # split the input sentences into words
    for token in doc:
        if token.text in tokenizer.word_index:
            sent_seq.append(tokenizer.word_index[token.text])

        # handle the unknown words error
        else:
            sent_seq.append(tokenizer.word_index['<unk>'])

    sent_seq = tf.expand_dims(sent_seq, 0)
    # predict the category of input sentences
    pred = model(sent_seq)

    pred_class = np.argmax(pred.numpy(), axis=1)
    
    # choice a random response for predicted sentence
    return random.choice(intent_doc[trg_index_word[pred_class[0]]]), trg_index_word[pred_class[0]]

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('chat.html')

@app.route("/ask", methods=['POST'])
def ask():
    message = request.form['messageText']
    print(message)
    while True:
        if message == "quit":
            exit()
        else:
            bot_response, intent_type = response(message)
            print(bot_response)
            return jsonify({'status':'OK','answer':bot_response, 'intent_type':intent_type})

if __name__ == '__main__':
    app.run()