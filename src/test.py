import torch
import torch.nn as nn
from text_preprocessing import clean_text, Vocab
import config
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from data import LSTMDataset
from model import LSTMClassifier
from engine import loss_fn
import pickle
import os


print("Device : ", config.device)
df = pd.read_csv(config.INPUT_DATA_PATH).fillna("none")
df.sentiment = df.sentiment.apply(
    lambda x: 1 if x == "positive" else 0
)
# df = df.sample(100, random_state=18)
vocab = Vocab()
for index, row in tqdm(df.iterrows()):
    vocab.addSentence(clean_text(row["review"]))
with open(config.vocab_file, "wb") as vocab_file:
    pickle.dump(vocab, vocab_file)

print("pickle file dumped")
print(os.path.getsize(config.vocab_file))
"""
df_train, df_test = train_test_split(df, test_size=0.1, stratify=df.sentiment.values)

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

train_dataset = LSTMDataset(df_train.review, df_train.sentiment, vocab)
test_dataset = LSTMDataset(df_test.review, df_test.sentiment, vocab)

train_dataloader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=False)

example = next(iter(train_dataloader))
# print(example[0])
# print(example[1])
example_input = example[0].to(device=config.device)
example_out = example[1].to(device=config.device)


model = LSTMClassifier(emb_size=64, vocab=vocab, hidden_size=128, seq_len=config.MAX_LEN, batch_size=config.TRAIN_BATCH_SIZE)
model.to(device=config.device)

out = model(example_input)
# print(out)
print(out.shape)
# print(example_out)
print(example_out.shape)

loss = loss_fn(out, example_out)
print("Loss function : ", loss)
print(loss.item())
"""