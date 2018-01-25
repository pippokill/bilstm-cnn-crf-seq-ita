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
from keras.layers.pooling import GlobalAveragePooling1D
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
    def themodel(embedding_weights, dictonary_size, MAX_SEQUENCE_LENGTH,MAX_CHARACTER_LENGTH, alfabeth_size, feature_size, tags,sentiment):    
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
       themodel2= Dropout(0.5)(themodel)
       themodel1= TimeDistributed(Dense(tags))(themodel2)
       crf=ChainCRF()
       crf_output= crf(themodel1)
       output=[]
       output.append(crf_output)
       if sentiment:
           out_d= Bidirectional(LSTM(Model.lstm_size))(themodel2)
           out_d= Dropout(0.5)(out_d)
           #out_d= GlobalAveragePooling1D(name='outd')(themodel2)
           out1= Dense(1,activation='sigmoid', name='out1' )(out_d)
           out2= Dense(1,activation='sigmoid', name='out2' )(out_d)
           out3= Dense(1,activation='sigmoid', name='out3' )(out_d)
           out4= Dense(1,activation='sigmoid', name='out4' )(out_d)
           out5= Dense(1,activation='sigmoid', name='out5' )(out_d)
           out6= Dense(1,activation='sigmoid', name='out6' )(out_d)
           output.append(out1)
           output.append(out2)
           output.append(out3)
           output.append(out4)
           output.append(out5)
           output.append(out6)
       input_list=[]
       input_list.append(word_input)
       if Model.embedding_char:
           input_list.append(character_input)
       if Model.features:
           input_list.append(feature_input)
       model= KerasModel(inputs=input_list,outputs=output)
       return crf,model
    
    def __init__(self, features, feature_dim,embed_char,grad_clipping,char_dim,filters,lstm_size,window,learning_alghoritm, learning_rate, decay,embedding_weights, dictonary_size, MAX_SEQUENCE_LENGTH,MAX_CHARACTER_LENGTH, alfabeth_size, feature_size, tags,sentiment):
       Model.EMBEDDING_WORD_DIM=embedding_weights.shape[1]
       Model.EMBEDDING_CHAR_DIM=char_dim
       Model.EMBEDDING_FEATURE_DIM= feature_dim
       Model.N_FILTERS=filters
       Model.lstm_size=lstm_size
       Model.window=window
       Model.embedding_char=embed_char
       Model.features=features
       crf, self.model=Model.themodel(embedding_weights, dictonary_size, MAX_SEQUENCE_LENGTH,MAX_CHARACTER_LENGTH, alfabeth_size,feature_size, tags,sentiment)
        # , metrics=[sparse_categorical_accuracy] 
       optimizer=None
       if learning_alghoritm=='Adadelta':
           optimizer=Adadelta(clipvalue=grad_clipping)
       elif learning_alghoritm=='Adagrad':
           optimizer=Adagrad(clipvalue=grad_clipping)
       elif learning_alghoritm=='SGD':
           optimizer=SGD(lr=learning_rate,decay=decay, momentum=0.9, clipvalue=grad_clipping)
       else:
           optimizer=learning_alghoritm
       losses=[crf.sparse_loss]
       metrics=[sparse_categorical_accuracy]
       if sentiment:
           losses.append('binary_crossentropy')
           losses.append('binary_crossentropy')
           losses.append('binary_crossentropy')
           losses.append('binary_crossentropy')
           losses.append('binary_crossentropy')
           losses.append('binary_crossentropy')
           metrics.append('accuracy')
       self.model.compile(loss=losses,optimizer=optimizer,metrics=metrics)
    
    
    
   
