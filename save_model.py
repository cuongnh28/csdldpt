from os import listdir
from os.path import isfile, join
from tika import parser
titles = [f for f in listdir("data") if isfile(join("data", f))] #De lay titles la ten file.
# luu cac doan van ban thanh mot list
documents = []
for file in titles:
    raw = parser.from_file('data/' + file)
    documents.append(raw['content'])

from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords

nltk.download('punkt')
stop_words = set(stopwords.words('english'))
# love loving loves -> love
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

class LemmaTokenizer:
    """
    Interface to the WordNet lemmatizer from nltk
    """
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]

# Demonstrate the job of the tokenizer

tokenizer=LemmaTokenizer()

token_stop = tokenizer(' '.join(stop_words))
vectorizer = TfidfVectorizer(stop_words=token_stop, tokenizer=tokenizer)

vectors = vectorizer.fit_transform(documents) # Trich xuat dac trung.

import pickle

with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)
with open('vectors.pkl', 'wb') as file:
    pickle.dump(vectors, file)

