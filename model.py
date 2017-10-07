#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 18:44:04 2017

@author: pc-plg
"""

from keras.models import Model as KerasModel
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import concatenate
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.wrappers import TimeDistributed
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import ChainCRF
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adadelta
from keras.optimizers import Adagrad
from keras.optimizers import SGD
from keras.initializers import RandomUniform
from keras.metrics import sparse_categorical_accuracy
from math import sqrt




class Model(object):
    EMBEDDING_WORD_DIM= 300
    EMBEDDING_CHAR_DIM= 30
    EMBEDDING_FEATURE_DIM= 40
    N_FILTERS=30
    window=3
    embedding_char= True
    features=False
    lstm_size=200
    def themodel(embedding_weights, dictonary_size, MAX_SEQUENCE_LENGTH,MAX_CHARACTER_LENGTH, alfabeth_size, feature_size, tags):    
       word_input= Input((MAX_SEQUENCE_LENGTH,))
       embed_out=Embedding(dictonary_size+1,
                                Model.EMBEDDING_WORD_DIM, 
                                weights=[embedding_weights],
                                        input_length=MAX_SEQUENCE_LENGTH, name='word_embedding')(word_input)
       word=TimeDistributed(Flatten())(embed_out)
       conc_list=[]
       conc_list.append(word)
       if Model.embedding_char:
           character_input=Input((MAX_SEQUENCE_LENGTH,MAX_CHARACTER_LENGTH,))
           embed_char_out=TimeDistributed(Embedding(alfabeth_size+1,Model.EMBEDDING_CHAR_DIM,
                                                    embeddings_initializer=RandomUniform(-sqrt(3/Model.EMBEDDING_CHAR_DIM),sqrt(3/Model.EMBEDDING_CHAR_DIM))), name='char_embedding')(character_input)
           dropout= Dropout(0.5)(embed_char_out)
           conv1d_out= TimeDistributed(Convolution1D(kernel_size=Model.window, filters=Model.N_FILTERS, padding='same',activation='tanh', strides=1))(dropout)
           maxpool_out=TimeDistributed(MaxPooling1D(MAX_CHARACTER_LENGTH))(conv1d_out)
           char= TimeDistributed(Flatten())(maxpool_out)
           conc_list.append(char)
       if Model.features:
           feature_input=Input((MAX_SEQUENCE_LENGTH,))
           featu= Embedding(feature_size, Model.EMBEDDING_FEATURE_DIM, input_length=MAX_SEQUENCE_LENGTH, name='feature_embedding')(feature_input)
           conc_list.append(featu)
       if Model.embedding_char or Model.features: 
           themodel= concatenate(conc_list)
       else:
           themodel=embed_out
       themodel= Dropout(0.5)(themodel)
       themodel= Bidirectional(LSTM(Model.lstm_size,return_sequences=True))(themodel)
       themodel= Dropout(0.5)(themodel)
       themodel= TimeDistributed(Dense(tags))(themodel)
       crf=ChainCRF()
       output= crf(themodel)
       input_list=[]
       input_list.append(word_input)
       if Model.embedding_char:
           input_list.append(character_input)
       if Model.features:
           input_list.append(feature_input)
       model= KerasModel(inputs=input_list,outputs=output)
       return crf,model
    
    def __init__(self, features, feature_dim,embed_char,grad_clipping,char_dim,filters,lstm_size,window,learning_alghoritm, learning_rate, decay,embedding_weights, dictonary_size, MAX_SEQUENCE_LENGTH,MAX_CHARACTER_LENGTH, alfabeth_size, feature_size, tags):
       Model.EMBEDDING_WORD_DIM=embedding_weights.shape[1]
       Model.EMBEDDING_CHAR_DIM=char_dim
       Model.EMBEDDING_FEATURE_DIM= feature_dim
       Model.N_FILTERS=filters
       Model.lstm_size=lstm_size
       Model.window=window
       Model.embedding_char=embed_char
       Model.features=features
       crf, self.model=Model.themodel(embedding_weights, dictonary_size, MAX_SEQUENCE_LENGTH,MAX_CHARACTER_LENGTH, alfabeth_size,feature_size, tags)
        # , metrics=[sparse_categorical_accuracy] 
       optimizer=None
       if learning_alghoritm=='Adadelta':
           optimizer=Adadelta(clipvalue=grad_clipping)
       elif learning_alghoritm=='Adagrad':
           optimizer=Adagrad(clipvalue=grad_clipping)
       else:
           optimizer=SGD(lr=learning_rate,decay=decay, momentum=0.9, clipvalue=grad_clipping)
       self.model.compile(loss=crf.sparse_loss, optimizer=optimizer,metrics=[sparse_categorical_accuracy])
    
    
    
   
