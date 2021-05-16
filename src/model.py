import transformers
import torch.nn as nn
import torch


class BERTClassificationModel(nn.Module):
    def __init__(self):
        super(BERTClassificationModel, self).__init__()
        self.bert_model = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        _, out2 = self.bert_model(ids, attention_mask=mask, token_type_ids=token_type_ids).values()
        do = self.bert_drop(out2)
        output = self.out(do)
        return output


class LSTMClassifier(nn.Module):
    def __init__(self, vocab, seq_len, emb_size, hidden_size, num_layers=3):
        super(LSTMClassifier, self).__init__()
        self.vocab = vocab
        self.emb_layer = nn.Embedding(vocab.n_words, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.linear1 = nn.Linear(seq_len*hidden_size, 128)
        nn.init.kaiming_normal_(self.linear1.weight.data)
        self.linear2 = nn.Linear(128, 32)
        nn.init.kaiming_normal_(self.linear2.weight.data)
        self.linear3 = nn.Linear(32, 1)
        nn.init.kaiming_normal_(self.linear3.weight.data)
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.2)
        self.drop3 = nn.Dropout(0.3)
        self.norm1 = nn.BatchNorm1d(seq_len)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.emb_layer(x)
        x, (_, _) = self.lstm(x)
        x = self.drop3(self.norm1(x))
        x = x.reshape(batch_size, -1)
        x = self.drop2(self.linear1(x))
        x = self.drop1(self.linear2(x))
        x = self.linear3(x)
        return x
