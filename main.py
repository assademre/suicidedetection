import pickle
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import re
import json

SAMPLE_NUMBER: int = 20000
N_GRAM_RANGE: int = 2

with open('corpus.json') as f:
    corpus = json.load(f)
model = pickle.load(open('model.pkl', 'rb'))
cv = CountVectorizer(max_features=SAMPLE_NUMBER, ngram_range=(1,N_GRAM_RANGE))
X = cv.fit_transform(corpus).toarray()

def suicide_prediction(text:str)->str:
    text = re.sub(pattern='[^a-zA-Z]',repl=' ', string=text)
    text = text.lower()
    text_words = text.split()
    text_words = [word for word in text_words if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    stemmed_form = [ps.stem(word) for word in text_words]
    stemmed_form = ' '.join(stemmed_form)
    temp = cv.transform([stemmed_form]).toarray()
    return model.predict(temp)[0]


