#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:49:17 2022

@author: root
"""

#%%
import pandas as pd
import numpy as np
import re 
from nltk.tokenize import word_tokenize,TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
import gensim 
import logging

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv1D,Dropout,MaxPooling1D,GlobalMaxPool1D,Activation,Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#%%  Used Function

def Clean_sentences(sent):
    stemmer = PorterStemmer()
    lemmatizer=WordNetLemmatizer()
    sent= re.sub('[^a-zA-Z]', ' ', sent)
    tokens=word_tokenize(sent)
    tokens=[lemmatizer.lemmatize(stemmer.stem(word.lower())) for word in tokens if word.lower() not in stopwords.words('english')and len(word)>1 and word.isalnum()]
    return tokens
    
    
#%%  loading cleaning data

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

dataset=pd.read_csv('TrainData.csv',names=['Types','Sentences'])
dataset.info()

Types = dataset['Types'].values
Sentences = dataset['Sentences'].values
worldlist=[]

for Sentence in Sentences :
        worldlist.append(Clean_sentences(Sentence))

print(np.max([len(ligne) for ligne in worldlist]))

#%%    build word2vect model

w2v_model = gensim.models.Word2Vec(worldlist, vector_size=300, window=5, min_count=1, workers=10)
w2v_model.train(worldlist,total_examples=len(worldlist),epochs=10)
w2v_model.save("word2vec.model")    

w2v_model=gensim.models.Word2Vec.load('word2vec.model')
print(w2v_model,len(w2v_model.wv.vectors),sep='\n')


#%% preprocessing data 

Keras_tk = Tokenizer(len(w2v_model.wv.vectors))
Keras_tk.fit_on_texts(Sentences)

X_train_sequence = Keras_tk.texts_to_sequences(Sentences)

MAX_LENGTH = np.max([len(ligne) for ligne in worldlist])

X_train_sequence  = pad_sequences(X_train_sequence, MAX_LENGTH )

print("Vocabulary size={}".format(len(Keras_tk.word_index)))

#%% Encoding Labels

Encodelabels=LabelEncoder()

Y= Encodelabels.fit_transform(Types)
Y= to_categorical(Y)

#%%  Split train And Test data

X_train, X_test, Y_train, Y_test = train_test_split(np.array(X_train_sequence), Y, train_size=0.85, stratify=Y)

X_train.shape, X_test.shape
Y_train.shape, Y_test.shape

#%% Build CNN model using keras

Keras_model = Sequential()
Keras_model.add(Embedding(len(Keras_tk.word_index), 400, input_length=MAX_LENGTH))
Keras_model.add(Conv1D(filters=64, kernel_size=16, padding='same', activation='relu'))
Keras_model.add(MaxPooling1D(pool_size=4))

Keras_model.add(Conv1D(filters=32, kernel_size=16, padding='same', activation='relu'))
Keras_model.add(MaxPooling1D(pool_size=4))
Keras_model.add(Flatten())
Keras_model.add(Dense(256, activation='relu'))
Keras_model.add(Dense(5, activation='softmax'))
Keras_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
Keras_model.summary()


#%% training

#Keras_model.fit(X_train, Y_train,validation_split=0.1,epochs=8,batch_size=100,verbose=1)
Keras_model.fit(X_train, Y_train, batch_size=128, epochs=4, validation_data=(X_test, Y_test))
Accuracy = Keras_model.evaluate(X_test, Y_test)
print("Accuracy: %.2f%%" % (Accuracy[1]*100))

#Keras_model.save('TextClassification_95.33%.h5')

#%%

predict = Keras_model.predict(X_test).ravel()

