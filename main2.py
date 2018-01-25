#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 15:11:52 2017

@author: pc-plg
"""

import data_processor
#from model import Model
import keras
import csv
from model2 import Model
import numpy as np   
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
from utils import iob_iobes, iob2, tag2index, check_tag_scheme,ConllevalCallback, run_conlleval, SentiPolcEval
import argparse
import sys

parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument('--task', type=str, default='NER')
parser.add_argument('--tag_scheme', type=str, default='IOBES', choices=['IOB2', 'IOBES'])
parser.add_argument('--train_path', type=str)
parser.add_argument('--dev_path', type=str)
parser.add_argument('--test_path', type=str)
parser.add_argument('--embed_path', type=str)
parser.add_argument('--embed_char', type=int, default=1)
parser.add_argument('--lowerword', type=int, default=1, help="All words in dataset-> lowercase")
parser.add_argument('--char_dim', type=int, default=30)
parser.add_argument('--word_dim', type=int, default=300)
parser.add_argument('--word2vecBINARY', type=int, default=0)
parser.add_argument('--label_column', type=int, default=3)
parser.add_argument('--word_column', type=int, default=0)
parser.add_argument('--embedding', type=str, default='word2vec', choices=['word2vec','glove','senna','random'])
parser.add_argument('--dataset_encoding', type=str, default='utf-8')
parser.add_argument('--embedding_encoding', type=str, default='utf-8')
parser.add_argument('--n_filters', type=int, default=30)
parser.add_argument('--window_size', type=int, default=3)
parser.add_argument('--lstm_size', type=int, default=200)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--learning_alghoritm', type=str, choices=['Adadelta', 'Adagrad', 'SGD','adam'], default='Adadelta')
parser.add_argument('--learning_rate', type=float, default=0.015)
parser.add_argument('--decay', type=float, default=0.05)
parser.add_argument('--grad_clipping', type=float, default=5.0)
parser.add_argument('--feature_column', type=int, default=2)
parser.add_argument('--features', type=int, default=0)
parser.add_argument('--log', type=str)
parser.add_argument('--log2', type=str)
parser.add_argument('--log3', type=str)
parser.add_argument('--log4', type=str)
parser.add_argument('--log5', type=str)
parser.add_argument('--log6', type=str)
parser.add_argument('--senti_train_path', type=str)
parser.add_argument('--senti_test_path', type=str)
parser.add_argument('--sentiment', type=int, default=0)
parser.add_argument('--k_fold', type=int, default=0)
parser.add_argument('--k_n', type=int, default=1)

args = parser.parse_args()
task=args.task
tag_scheme=args.tag_scheme
train_path=args.train_path
dev_path=args.dev_path
test_path=args.test_path
embedding_path=args.embed_path
embed_char=args.embed_char
char_dim=args.char_dim
word_dim=args.word_dim
word2vecBINARY=args.word2vecBINARY
label_column=args.label_column
word_column=args.word_column
embedding=args.embedding
dataset_encoding=args.dataset_encoding
embedding_encoding=args.embedding_encoding
n_filters=args.n_filters
window=args.window_size
lstm_size=args.lstm_size
epochs=args.epochs
batch=args.batch_size
learning_alghoritm=args.learning_alghoritm
learning_rate=args.learning_rate
decay=args.decay
grad_clipping=args.grad_clipping
lowerword=args.lowerword
feature_column=args.feature_column
features=args.features
log=args.log
log2=args.log2
log3=args.log3
log4=args.log4
log5=args.log5
log6=args.log6
sentiment=args.sentiment
senti_train_path=args.senti_train_path
senti_test_path=args.senti_test_path
k_fold=args.k_fold
k_n=args.k_n

task='POS'
train_path='/home/pc-plg/Scrivania/AnacondaProjects/tesi4/dataset/twit_ita.train'
dev_path='/home/pc-plg/Scrivania/AnacondaProjects/tesi4/dataset/twit_ita.train'
test_path='/home/pc-plg/Scrivania/AnacondaProjects/tesi4/dataset/twit_ita.test'
embedding_path='/home/pc-plg/Scrivania/AnacondaProjects/tesi4/dataset/w2v.300.100.all.vec'
embedding_encoding='ISO-8859-15'
sentiment=1
label_column=1
senti_train_path='/home/pc-plg/Scrivania/tesi4/makedata/senti.train'
senti_test_path='/home/pc-plg/Scrivania/tesi4/makedata/senti.test'
log='log'
log2='log2'
log3='log3'
log4='log4'
log5='log5'
log6='log6'
learning_alghoritm='Adagrad'
epochs=80


fine_tune=True
oov='embedding'
X_train, Y_train, mask_train, X_dev, Y_dev, mask_dev, X_test, Y_test, mask_test, \
    embedd_table, label_alphabet, word_alphabet, feature_alphabet, \
    C_train, C_dev, C_test, char_embedd_table, char_alphabet, F_train, F_dev, F_test= data_processor.load_dataset_sequence_labeling(train_path, dev_path,
                                                                                              test_path,features=features,word_column=word_column,
                                                                                              label_column=label_column, feature_column= feature_column,label_name=task,
                                                                                              oov=oov,
                                                                                              fine_tune=fine_tune,
                                                                                              embedding=embedding,
                                                                                              embedding_path=embedding_path,
                                                                                              use_character=True,dataset_encoding=dataset_encoding,
                                                                                              embedding_encoding=embedding_encoding, word2vecBINARY=word2vecBINARY, embedd_dims=word_dim, lowerword=lowerword, char_dim=char_dim)
index_to_label=[]
index_to_token=[]
index_to_token.append('PAD')
index_to_label.append('PAD')
#embeddings_metadata= {'word_embedding':'/home/pc-plg/AnacondaProjects/tesi4/metadata_tokens.tsv', 'char_embedding':'/home/pc-plg/AnacondaProjects/tesi4/metadata_characters.tsv'}
#tokensfile  = open(embeddings_metadata['word_embedding'], "w");
#tokenswriter = csv.writer(tokensfile, delimiter=',')

#charsfile  = open(embeddings_metadata['char_embedding'], "w");
#charswriter = csv.writer(charsfile,delimiter=',')

#tokenswriter.writerow([index_to_token[0]]);
#charswriter.writerow(['PAD']);

#for char, index in char_alphabet.iteritems():
#    charswriter.writerow([char]);
for word, index in word_alphabet.iteritems():
    index_to_token.append(word)
#    tokenswriter.writerow([word]);
for word, index in label_alphabet.iteritems():
    index_to_label.append(word)

 
#tokensfile.close()
#charsfile.close()
word_alphabet.close()
label_alphabet.close()

if(task=='NER'):
        if(tag_scheme=='IOBES'):
            index_to_label= check_tag_scheme(index_to_label)
        label_to_index= tag2index(index_to_label)
        i=0
        for x in [Y_train, Y_dev, Y_test]:
            for example in x:
                x[i]=iob2(example,index_to_label,label_to_index)
                if(tag_scheme=='IOBES'):
                    iob_iobes(x[i], index_to_label,label_to_index)
                i+=1
            i=0                 
    
if embed_char:
    X_tr=[X_train,C_train]
    X_de=[X_dev,C_dev]
    X_te=[X_test,C_test]
    if features:
        X_tr.append(F_train)
        X_de.append(F_dev)
        X_te.append(F_test)
else:
    if features:
         X_tr=[X_train, F_train]
         X_de=[X_dev, F_dev]
         X_te=[X_test, F_test]
    X_tr=X_train
    X_de=X_dev
    X_te=X_test

Y_train=np.expand_dims(Y_train,-1)
Y_dev=np.expand_dims(Y_dev,-1)
Y_test=np.expand_dims(Y_test,-1) 

logger = open(log,'w')
logger2 = open(log2,'w')
logger3 = open(log3,'w')
logger4 = open(log4,'w')
logger5 = open(log5,'w')
logger6 = open(log6,'w')

if features:
    feature_size=feature_alphabet.size()
else:
    feature_size=0

model = Model(features=features, feature_dim=40, embed_char=embed_char,grad_clipping=grad_clipping,char_dim=char_dim,filters=n_filters,lstm_size=lstm_size,window=window,learning_alghoritm=learning_alghoritm, learning_rate=learning_rate, decay=decay,embedding_weights=embedd_table, dictonary_size=len(index_to_token)-1, MAX_SEQUENCE_LENGTH=X_train.shape[1], MAX_CHARACTER_LENGTH=C_train.shape[2], alfabeth_size=char_embedd_table.shape[0]-1,feature_size=feature_size,tags=len(index_to_label),sentiment=sentiment)

#tensorboard=keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=1, embeddings_layer_names=['word_embedding','char_embedding'], embeddings_metadata=embeddings_metadata)

if sentiment:
    subj=np.array([])
    opos=np.array([])
    oneg=np.array([])
    iro=np.array([])
    lpos=np.array([])
    lneg=np.array([])
    top=np.array([])
    subj2=np.array([])
    opos2=np.array([])
    oneg2=np.array([])
    iro2=np.array([])
    lpos2=np.array([])
    lneg2=np.array([])
    top2=np.array([])
    train=[subj,opos,oneg,iro,lpos,lneg,top]
    test=[subj2,opos2,oneg2,iro2,lpos2,lneg2,top2]
    train_test=[train,test]
    paths=[senti_train_path,senti_test_path]
    i=0
    j=0
    for path in paths:
        f_senti=open(path)
        tupla=[]
        for line in f_senti:
            if line == 'ERROR KEY\n':
                tupla.append(j)
            else:
                line= line.strip()
                line= line.split()
                train_test[i][0]=np.append(train_test[i][0],int(line[0]))
                train_test[i][1]=np.append(train_test[i][1],int(line[1]))
                train_test[i][2]=np.append(train_test[i][2],int(line[2]))
                train_test[i][3]=np.append(train_test[i][3],int(line[3]))
                train_test[i][4]=np.append(train_test[i][4],int(line[4]))
                train_test[i][5]=np.append(train_test[i][5],int(line[5]))
                train_test[i][6]=np.append(train_test[i][6],int(line[6]))
            j=j+1
        if i==0:
            k=0
            for x in X_tr:
                X_tr[k]= np.delete(X_tr[k], tuple(tupla), axis=0)
                k=k+1
            Y_train=np.delete(Y_train, tuple(tupla), axis=0)
            k=0
            for x in X_de:
                X_de[k]= np.delete(X_de[k], tuple(tupla), axis=0)
                k=k+1
            Y_dev=np.delete(Y_dev, tuple(tupla), axis=0)
        else:
            k=0
            for x in X_te:
                X_te[k]= np.delete(X_te[k], tuple(tupla), axis=0)
                k=k+1
            Y_test=np.delete(Y_test, tuple(tupla), axis=0)
        i=i+1
        j=0
        f_senti.close()
    new_train=[]
    new_test=[]
    new_train.append(Y_train)
    for x in train[0:len(train)-1]:
        new_train.append(x)
    new_test.append(Y_test)
    for x in test[0:len(test)-1]:
        new_test.append(x)

if k_fold:
    part=int(len(X_tr[0])/5)
    begin=int(0+(part*(k_n-1)))
    end=int(part+(part*(k_n-1)))
    X_dev= X_tr[0][begin:end]
    C_dev= X_tr[1][begin:end]
    if features:
        F_dev= X_tr[2][begin:end]
    Y_dev= Y_train[begin:end]
    dev=[Y_dev]
    dev.append(train[0][begin:end])
    dev.append(train[1][begin:end])
    dev.append(train[2][begin:end])
    dev.append(train[3][begin:end])
    dev.append(train[4][begin:end])
    dev.append(train[5][begin:end])
    new_dev=dev
    X_de=[X_dev,C_dev]
    if features:
        X_de.append(F_dev)
    tr1=[]
    tr2=[]
    tr3=[]
    y=[]
    y2=[]
    y3=[]
    y4=[]
    y5=[]
    y6=[]
    y7=[]
    i=0
    while i != len(Y_train):
        if i< begin or i>=end:
            tr1.append(X_tr[0][i])
            tr2.append(X_tr[1][i])
            if features:
                tr3.append(X_tr[2][i])
            y.append(Y_train[i])
            y2.append(train[0][i])
            y3.append(train[1][i])
            y4.append(train[2][i])
            y5.append(train[3][i])
            y6.append(train[4][i])
            y7.append(train[5][i])
        i=i+1
    tr1=np.array(tr1)
    tr2=np.array(tr2)
    tr3=np.array(tr3)
    y=np.array(y)
    y2=np.array(y2)
    y3=np.array(y3)
    y4=np.array(y4)
    y5=np.array(y5)
    y6=np.array(y6)
    y7=np.array(y7)
    X_tr[0]=tr1
    X_tr[1]=tr2
    if features:
        X_tr[2]=tr3
    new_train=[y,y2,y3,y4,y5,y6,y7]
else:
    new_dev=new_train
        

conlleval= ConllevalCallback(logger,X_tr,new_train[0],index2word=index_to_token, index2chunk= index_to_label, batch_size=batch, task=task, tag_scheme=tag_scheme)
conlleval2= ConllevalCallback(logger2,X_de,new_dev[0],index2word=index_to_token, index2chunk= index_to_label, batch_size=batch, task=task, tag_scheme=tag_scheme) 
conlleval3= ConllevalCallback(logger3,X_te,Y_test,index2word=index_to_token, index2chunk= index_to_label, batch_size=batch, task=task, tag_scheme=tag_scheme) 
sentipolc= SentiPolcEval(out=1,logger=logger6,X_test=X_te,y_test=new_test[1],index2word=index_to_token, index2chunk= index_to_label, batch_size=batch, task='SUBJ', tag_scheme=tag_scheme) 
sentipolc2= SentiPolcEval(out=[2,3],logger=logger6,X_test=X_te,y_test=[new_test[2],new_test[3]],index2word=index_to_token, index2chunk= index_to_label, batch_size=batch, task='POL', tag_scheme=tag_scheme) 
sentipolc3= SentiPolcEval(out=4,logger=logger6,X_test=X_te,y_test=new_test[4],index2word=index_to_token, index2chunk= index_to_label, batch_size=batch, task='IRO', tag_scheme=tag_scheme) 
sentipolc4= SentiPolcEval(out=[5,6],logger=logger6,X_test=X_te,y_test=[new_test[5],new_test[6]],index2word=index_to_token, index2chunk= index_to_label, batch_size=batch, task='LITER', tag_scheme=tag_scheme) 

sentipolc5= SentiPolcEval(out=1,logger=logger4,X_test=X_tr,y_test=new_train[1],index2word=index_to_token, index2chunk= index_to_label, batch_size=batch, task='SUBJ', tag_scheme=tag_scheme) 
sentipolc6= SentiPolcEval(out=[2,3],logger=logger4,X_test=X_tr,y_test=[new_train[2],new_train[3]],index2word=index_to_token, index2chunk= index_to_label, batch_size=batch, task='POL', tag_scheme=tag_scheme) 
sentipolc7= SentiPolcEval(out=4,logger=logger4,X_test=X_tr,y_test=new_train[4],index2word=index_to_token, index2chunk= index_to_label, batch_size=batch, task='IRO', tag_scheme=tag_scheme) 
sentipolc8= SentiPolcEval(out=[5,6],logger=logger4,X_test=X_tr,y_test=[new_train[5],new_train[6]],index2word=index_to_token, index2chunk= index_to_label, batch_size=batch, task='LITER', tag_scheme=tag_scheme) 

sentipolc9= SentiPolcEval(out=1,logger=logger5,X_test=X_de,y_test=new_dev[1],index2word=index_to_token, index2chunk= index_to_label, batch_size=batch, task='SUBJ', tag_scheme=tag_scheme) 
sentipolc10= SentiPolcEval(out=[2,3],logger=logger5,X_test=X_de,y_test=[new_dev[2],new_dev[3]],index2word=index_to_token, index2chunk= index_to_label, batch_size=batch, task='POL', tag_scheme=tag_scheme) 
sentipolc11= SentiPolcEval(out=4,logger=logger5,X_test=X_de,y_test=new_dev[4],index2word=index_to_token, index2chunk= index_to_label, batch_size=batch, task='IRO', tag_scheme=tag_scheme) 
sentipolc12= SentiPolcEval(out=[5,6],logger=logger5,X_test=X_de,y_test=[new_dev[5],new_dev[6]],index2word=index_to_token, index2chunk= index_to_label, batch_size=batch, task='LITER', tag_scheme=tag_scheme) 
 
#plot_model(model.model,"model.png",show_shapes=True)

class EarlyStopping2(EarlyStopping):
        def __init__(self,monitor,patience,verbose):
                super(EarlyStopping2,self).__init__(monitor=monitor,patience=patience,verbose=verbose)
        def on_train_end(self, logs=None):
                super(EarlyStopping2,self).on_train_end(logs)
                print('EXPERIMENT '+log[0:len(log)-6]+' FINAL DEV LOSS '+ str(logs.get(self.monitor)))
                                

stop= EarlyStopping2(monitor='val_loss', patience=10, verbose=1)   
model.model.fit(X_tr, new_train, validation_data=(X_de, new_dev),epochs=epochs, batch_size=batch, callbacks=[conlleval,conlleval2,conlleval3,sentipolc,sentipolc2,sentipolc3,sentipolc4,stop], verbose=1)

logger.close()
logger2.close()
logger3.close()
logger4.close()
logger5.close()
logger6.close()
