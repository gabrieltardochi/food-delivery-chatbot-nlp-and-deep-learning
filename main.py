import random
import json
import torch
from src.model import NeuralNet
from src.utils import fix_sentence, bag_of_words, tokenize, stem
from flask import Flask, request, Markup, render_template, redirect


app = Flask(__name__)
msgs = []

# loading model and useful data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('data/intents.json', 'r') as json_data:
    intents = json.load(json_data)

mdl_path = f"model/mdl.pth"

mdl_dict = torch.load(mdl_path)

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
        return "Im not trained to understand that yet.. please try something else!"


@app.route('/')
def clean_chat():
    global msgs
    msgs = [Markup(f'<p class="pbot">Hi! I am Tob, a demo of a food delivery ChatBot. Come chat with me!</p>')]
    return render_template('home.html', msgs=msgs)


@app.route('/chatting', methods=['POST'])
def add_msg_and_predict():
    global msgs
    sentence = request.form['content']
    if not sentence:
        return render_template('list.html', msgs=msgs)
    msgs += [Markup(f'<p class="pother">{sentence}</p>')]

    answer = answer_raw_sentence(sentence, all_words, intents, tags, model, device)
    msgs += [Markup(f'<p class="pbot">{answer}</p>')]
    
    return render_template('home.html', msgs=msgs)


if __name__ == '__main__':
    app.run(debug=True)