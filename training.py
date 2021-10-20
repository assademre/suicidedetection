import pandas as pd
import nltk
import re
import json
from pickle import dump
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tqdm import tqdm

DATABASE_NAME: str = "Suicide_Detection.csv"
SAMPLE_NUMBER: int = 50000
TEST_SIZE: float = 0.20
N_GRAM_RANGE: int = 2

df = pd.read_csv(DATABASE_NAME)

mapping = {"non-suicide": 0, "suicide": 1}
df[:SAMPLE_NUMBER]['class'] = df[:SAMPLE_NUMBER]['class'].map(mapping)
df.isna().any()
df.drop('Unnamed: 0', axis=1, inplace=True)
corpus = []
ps = PorterStemmer()

for i in tqdm(range(0, SAMPLE_NUMBER)):
    text = re.sub(pattern='[^a-zA-Z]', repl=' ', string=df['text'][i])
    text = text.lower()
    words = text.split()
    text_words = [word for word in words if word not in set(stopwords.words('english'))]
    words = [ps.stem(word) for word in text_words]
    joined_words = ' '.join(words)
    corpus.append(joined_words)

df = pd.read_csv(DATABASE_NAME)[:SAMPLE_NUMBER]
cv = CountVectorizer(max_features=SAMPLE_NUMBER, ngram_range=(1,N_GRAM_RANGE))
X = cv.fit_transform(corpus).toarray()
y = df['class'].values

with open('corpus.json', 'w') as f:
    json.dump(corpus, f)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=0)
print(f'X_train size: {X_train.shape}, X_test size: {X_test.shape}')
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
dump(classifier, open('model.pkl', 'wb'))
y_prediction = classifier.predict(X_test)
score1 = accuracy_score(y_test, y_prediction)
print(f"Accuracy score is: {round(score1*100,2)}%")
