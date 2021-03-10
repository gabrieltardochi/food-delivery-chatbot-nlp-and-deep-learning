import random
import time
import json
import torch
from src.model import NeuralNet
from src.utils import fix_sentence, bag_of_words, tokenize, stem
from flask import Flask, request, Markup, render_template, session


app = Flask(__name__)
app.secret_key = "my_secret_key"

# loading model (on cpu for heroku) and useful data
device = torch.device('cpu')

with open('data/intents.json', 'r') as json_data:
    intents = json.load(json_data)

mdl_path = f"model/mdl.pth"

mdl_dict = torch.load(mdl_path, map_location=device)

input_size = mdl_dict["input_size"]
hidden_size = mdl_dict["hidden_size"]
output_size = mdl_dict["output_size"]
all_words = mdl_dict['all_words']
tags = mdl_dict['tags']
model_state = mdl_dict["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


def answer_raw_sentence(sentence, all_words, intents, tags, mdl, device):
    """
    passes raw sentence throught preprocessing pipe and fetch model answer
    """
    sentence = fix_sentence(sentence)
    sentence = tokenize(sentence)
    sentence = [stem(w) for w in sentence]

    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = mdl(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.6:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    else:
        return "Im not ready to understand that yet.. please try something else!"


@app.route('/')
def clean_chat():
    session["msgs"] = [Markup(f'<p class="pbot">Hi! I am Tob, a demo of a food delivery ChatBot. Come chat with me!</p>')]
    time.sleep(1)
    return render_template('home.html', msgs=session["msgs"])


@app.route('/chatting', methods=['POST'])
def add_msg_and_predict():
    session["this_sentence"] = request.form['content']
    if not session["this_sentence"]:
        time.sleep(1)
        return render_template('home.html', msgs=session["msgs"])
    session["msgs"] += [Markup(f'<p class="pother">{session["this_sentence"]}</p>')]

    session["this_answer"] = answer_raw_sentence(session["this_sentence"], all_words, intents, tags, model, device)
    session["msgs"] += [Markup(f'<p class="pbot">{session["this_answer"]}</p>')]
    time.sleep(1)
    return render_template('home.html', msgs=session["msgs"])


if __name__ == '__main__':
    app.run()