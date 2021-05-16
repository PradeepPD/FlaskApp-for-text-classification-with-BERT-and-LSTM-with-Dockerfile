import config
import transformers
import torch
from text_preprocessing import clean_text
from torch.utils.data import Dataset


class BERTDataset:
    def __init__(self, review, target):
        self.review = review
        self.target = target
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        review = str(self.review[idx])
        review = " ".join(review.split())

        inputs = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True
        )

        ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'target': torch.tensor(self.target[idx], dtype=torch.float)
        }


class LSTMDataset(Dataset):
    def __init__(self, review, target, vocab):
        super(LSTMDataset, self).__init__()
        self.review = review
        self.target = target
        self.vocab = vocab
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        review = clean_text(self.review[idx])
        review = [self.vocab.word2index.get(word, 0) for word in review]

        if len(review) < self.max_len:
            review = review + ([0] * (self.max_len - len(review)))
        elif len(review) > self.max_len:
            review = review[:self.max_len]

        return torch.tensor(review, dtype=torch.long), torch.tensor(self.target[idx], dtype=torch.float)

