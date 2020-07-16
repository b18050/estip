# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# pkl_builder.py

"""
Import Libraries
"""

import pickle
import re
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

"""
Tokenizer
"""

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stopwords.words('english')]
    # this will return the cleaned text
    return tokenized


vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)

"""
Classifier 
"""

clf = SGDClassifier(loss='log', random_state=1, max_iter=100)

def load_file(__path_to_file__):
    df = pd.read_csv(__path_to_file__, encoding='utf-8')
    return df


def build_model():
    
    X_train = df['review'].values
    y_train = df['sentiment'].values

    X_train = vect.transform(X_train)
    clf.fit(X_train, y_train)
    return clf

def classify(document,clf):
    X = vect.transform([document])
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    print(y)
    print("\n",proba)
    
df = load_file(r'___url_to_path___')
clf = build_model()
feedback = 'bad. Perfect'
classify(feedback,clf)

pickle.dump(stopwords.words('english'),open('stopwords.pkl', 'wb'),protocol=4)

pickle.dump(clf,open('classifier.pkl', 'wb'),protocol=4)

