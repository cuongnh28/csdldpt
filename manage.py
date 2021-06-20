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

from flask import Flask,render_template,request,Response
import search

app=Flask(__name__)

#route() decorators
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/search_tmp/', methods=['POST'])
def search_tmp():
    input_text = request.form['input_text']
    result = search.search(input_text)
    return render_template('Result.html', len=len(result), result=result)

@app.route('/read_file')
def read_file():
    file_name = request.args.get('file')
    f = open("data/"+file_name, "r")
    content = f.read()
    return Response(content, mimetype='text/plain')

if __name__=='__main__':
    app.run(debug=True)
