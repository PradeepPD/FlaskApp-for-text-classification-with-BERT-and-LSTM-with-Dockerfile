import config
from data import LSTMDataset, BERTDataset
import torch
import transformers
import engine
import pandas as pd
import numpy as np
from model import BERTClassificationModel, LSTMClassifier
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn import metrics
from text_preprocessing import clean_text
from text_preprocessing import Vocab
from tqdm import tqdm
from data import LSTMDataset
from torch import optim
import pickle


def run_bert():
    print("Device : ", config.device)
    df = pd.read_csv(config.INPUT_DATA_PATH).fillna("none")
    df.sentiment = df.sentiment.apply(
        lambda x : 1 if x == "positive" else 0
    )
    # df = df.sample(1000, random_state=18)

    df_train, df_test = train_test_split(df, test_size=0.1, stratify=df.sentiment.values)

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    train_dataset = BERTDataset(df_train.review.values, df_train.sentiment.values)
    train_dataloader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=4)

    valid_dataset = BERTDataset(df_test.review.values, df_test.sentiment.values)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.VALID_BATCH_SIZE , shuffle=False, num_workers=1)

    model = BERTClassificationModel()
    # param_optimizer = list(model.named_parameters())
    # no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # optimizer_parameters = [
    #     {'params' : [p for n, p in param_optimizer if not in no_decay]}
    # ]

    model.to(config.device)
    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = transformers.AdamW(model.parameters(), lr=0.0001)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps = num_train_steps)

    best_acc = 0
    for epoch in range(config.EPOCHS):
        print("Starting epoch {}".format(epoch+1))
        engine.train_fn(train_dataloader, model, optimizer, config.device, scheduler)
        outputs, targets = engine.eval_fn(valid_dataloader, model, config.device)
        outputs = np.array(outputs) >= 0.5
        eval_accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Epoch : {epoch+1}, accuracy : {eval_accuracy}")
        if eval_accuracy > best_acc:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_acc = eval_accuracy

def run_lstm():
    print("Device : ", config.device)
    df = pd.read_csv(config.INPUT_DATA_PATH).fillna("none")
    df.sentiment = df.sentiment.apply(
        lambda x : 1 if x == "positive" else 0
    )
    # df = df.sample(1000, random_state=18)
    vocab = Vocab()
    for index, row in tqdm(df.iterrows()):
        vocab.addSentence(clean_text(row["review"]))

    with open("vocab.pkl", "wb") as vocab_file:
        pickle.dump(vocab, vocab_file)

    df_train, df_test = train_test_split(df, test_size=0.1, stratify=df.sentiment.values)

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    train_dataset = LSTMDataset(df_train.review, df_train.sentiment, vocab)
    test_dataset = LSTMDataset(df_test.review, df_test.sentiment, vocab)

    train_dataloader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=False)

    model = LSTMClassifier(emb_size=config.emb_size, vocab=vocab, hidden_size=config.hidden_size, seq_len=config.MAX_LEN)
    model.to(device=config.device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=2)

    best_acc = 0
    for epoch in range(config.EPOCHS):
        print("Starting epoch {}".format(epoch+1))
        train_loss = engine.lstm_train_fn(train_dataloader, model, optimizer, config.device, scheduler)
        outputs, targets, valid_loss = engine.lstm_eval_fn(test_dataloader, model, config.device)
        outputs = np.array(outputs) >= 0.5
        eval_accuracy = metrics.accuracy_score(targets, outputs)
        print(f"\nEpoch : {epoch+1}, accuracy : {eval_accuracy}, Train loss : {train_loss}, Validation Loss : {valid_loss}")
        if eval_accuracy > best_acc:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_acc = eval_accuracy


if __name__ == "__main__":
    run_lstm()
    # print(config.INPUT_DATA_PATH)

