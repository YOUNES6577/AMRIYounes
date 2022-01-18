#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 11:30:48 2022

@author: root
"""

import pandas as pd
import numpy as np
import re 
from nltk.tokenize import word_tokenize,TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
import gensim 
import logging

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

#%%

def Clean_sentences(sent):
    stemmer = PorterStemmer()
    lemmatizer=WordNetLemmatizer()
    sent= re.sub('[^a-zA-Z]', ' ', sent)
    tokens=word_tokenize(sent)
    tokens=[stemmer.stem(word.lower()) for word in tokens if word.lower() not in stopwords.words('english') and word.isalnum()]
    return tokens

#%%

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

dataset=pd.read_csv('Testdata.csv',names=['ID','text'])
dataset.info()
dataset.head()

text = dataset['text'].values
worldlist=[]

for Sentence in text :
        worldlist.append(Clean_sentences(Sentence))
        
#%%
w2v_model=gensim.models.Word2Vec.load('word2vec.model')
MAX_LENGTH = 2194

Keras_tk = Tokenizer(len(w2v_model.wv.vectors))
Keras_tk.fit_on_texts(text)

X_test_sequence = Keras_tk.texts_to_sequences(text)
X_test_sequence  = pad_sequences(X_test_sequence, MAX_LENGTH )

#%%
Keras_model = load_model('TextClassification_95.33%.h5')
Probabilities=Keras_model.predict(X_test_sequence)

Class_Dict={0:'business',1:'entertainment',2:'sport',3:'tech',4:'politics'}
FinalResult={'ID':[],'Proba':[],'Class':[]}

for i in range(len(Probabilities)):
    maxProba=np.max(Probabilities[i])
    Class=[j for j in range(len(Probabilities[i])) if Probabilities[i][j]==maxProba][0]
    Class=Class_Dict[Class]
    FinalResult['ID'].append(i+1)
    FinalResult['Proba'].append(str("%.2f%%" % (maxProba*100)))
    FinalResult['Class'].append(Class)
    
#%%

df = pd.DataFrame({'ID': FinalResult['ID'],
                   'Proba': FinalResult['Proba'],
                   'Class':FinalResult['Class']
                   })

df.to_csv('YOUNES AMRI - Result.csv', index=False)
