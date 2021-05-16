import torch
import transformers

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 50
INPUT_DATA_PATH = "../data/IMDB_Dataset.csv"
MODEL_PATH = "model.bin"
TOKENIZER = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
app_device = torch.device('cpu')
vocab_file = "vocab.pkl"
emb_size = 64
hidden_size = 128