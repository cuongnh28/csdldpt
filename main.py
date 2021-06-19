from os import listdir
from os.path import isfile, join
titles = [f for f in listdir("data") if isfile(join("data", f))]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

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

# vectors = vectorizer.fit_transform(documents) # Trich xuat dac trung.

import pickle

vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
vectors = pickle.load(open('vectors.pkl', 'rb'))

search_terms = 'Declan Rice (born 14 January 1999) is an English professional footballer who plays as a defensive midfielder or centre-back for Premier League club West Ham United and the England national team.'

test_idf = vectorizer.transform([search_terms])
# print(vectors.toarray().shape)
# print(test_idf.toarray().shape)

cosine_similarities = linear_kernel(vectors, test_idf).flatten() #Tinh linear kerner.
document_scores = [item.item() for item in cosine_similarities[1:]]

score_titles = [(score, title) for score, title in zip(document_scores, titles)]

for score, title in (sorted(score_titles, reverse=True, key=lambda x: x[0])[:10]):
    print(f'{score:0.3f} \t {title}')

print("end")

