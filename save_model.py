from os import listdir
from os.path import isfile, join
titles = [f for f in listdir("data") if isfile(join("data", f))] #De lay titles la ten file.

# luu cac doan van ban thanh mot list
documents = []

for file in titles:
    f = open('data/'+file, 'r')
    documents.append(f.read())

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
# love loving loves -> love
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

class LemmaTokenizer:
    """
    Interface to the WordNet lemmatizer from nltk
    """
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`', '!', '#', '$',
                     '%', "'", '/', '-', '&', '<', '?', '...', '....', '..', "'in",
                     '*', '--', "-it", "-do", '‘', '’', '“', '•', '…', '”', '.—i',
                     '.—it', '.—oh', "'70s", "'do", "'dolph", "'em", "'forrester",
                     "'if", "'it", "'m", "'mrs", "'niel", "'pon", "'tell", "'way",
                     "'when", '(', ')', '-all', '.004', '.20', '—was',
                     '—well', '—were', '—what', '—where', '—which', '—while',
                     '—why', '—with', '—world', '—would', '—you']
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]
# Demonstrate the job of the tokenizer

tokenizer=LemmaTokenizer()

token_stop = tokenizer(' '.join(stop_words))
vectorizer = TfidfVectorizer(stop_words=token_stop, tokenizer=tokenizer) #de luu tu vung va tokenize
vectors = vectorizer.fit_transform(documents) # Trich xuat dac trung.
# print(vectorizer.get_feature_names())
# print(vectors)
import pickle

with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)
with open('vectors.pkl', 'wb') as file:
    pickle.dump(vectors, file)

