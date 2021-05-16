import flask
import pickle
import config
import torch
import time
from flask import Flask, request, render_template
from model import BERTClassificationModel, LSTMClassifier
from text_preprocessing import clean_text

app = Flask(__name__)


def sentence_prediction(sentence):
    review = str(sentence)
    review = " ".join(review.split())

    inputs = config.TOKENIZER.encode_plus(
        review,
        add_special_tokens=True,
        max_length=config.MAX_LEN,
        padding="max_length",
        truncation=True
    )

    ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0).to(config.app_device)
    token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long).unsqueeze(0).to(config.app_device)
    mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0).to(config.app_device)

    output = model(
        ids, mask, token_type_ids
    )
    prediction = torch.sigmoid(output).cpu().detach().numpy()
    return prediction[0][0]


def sentence_predict_lstm(review):
    # with open("vocab.pkl", "wb") as vocab_file:
    #     vocab = pickle.load(vocab_file)
    review = clean_text(review)
    review = [vocab.word2index.get(word, 0) for word in review]

    if len(review) < config.MAX_LEN:
        review = review + ([0] * (config.MAX_LEN - len(review)))
    elif len(review) > config.MAX_LEN:
        review = review[:config.MAX_LEN]

    review = torch.tensor(review, dtype=torch.long)
    review = review.unsqueeze(0).to(config.app_device)
    prediction = torch.sigmoid(model(review)).cpu().detach().numpy()
    return prediction[0][0]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def predict():
    start_time = time.time()
    sentence = request.form.get("sentence")
    # print(sentence)
    positive_prediction = round(sentence_predict_lstm(sentence), 3)
    # response = {"response": {
    #     'sentence': str(sentence),
    #     'positive': str(positive_prediction),
    #     'negative': str(round(1 - positive_prediction, 3)),
    #     'time taken': str(time.time() - start_time)
    # }}
    return render_template(
        'response.html',
        sentence=str(sentence),
        positive_sentiment = str(round(positive_prediction, 2)),
        negative_sentiment = str(round(1-positive_prediction, 2)),
        time = str(round(time.time()-start_time, 2))
    )


if __name__ == "__main__":
    # model = BERTClassificationModel()
    with open(config.vocab_file, "rb") as vocab_file:
        vocab = pickle.load(vocab_file)
    model = LSTMClassifier(emb_size=config.emb_size, vocab=vocab, hidden_size=config.hidden_size, seq_len=config.MAX_LEN)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.app_device))
    model.to(config.app_device)
    model.eval()
    # app.run(host='127.0.0.1', port='5000', debug=True)
    app.run(host='0.0.0.0', port='5000', debug=True)
