import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import string

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

repalce_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
st = PorterStemmer()

def clean_text(txt):
    txt = txt.translate(repalce_punctuation).lower()
    txt = [st.stem(lemmatizer.lemmatize(i)) for i in word_tokenize(txt) if i not in stop_words and str.isalpha(i)]
    return txt

class Vocab:
    def __init__(self):
        self.word2index = {"<unk>": 1}
        self.word2index = {"<PAD>": 0}
        self.word2count = {}
        self.index2word = {1: "<unk>"}
        self.index2word = {0: "<PAD>"}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence:
            self._addWord(word)

    def _addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

