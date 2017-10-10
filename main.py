#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2017 Pierpaolo Basile, Pierluigi Cassotti

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import data_processor
from model import Model
import keras
import csv
import numpy as np
from keras.utils.vis_utils import plot_model
from utils import iob_iobes, iob2, tag2index, check_tag_scheme,ConllevalCallback, run_conlleval
import argparse

parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument('--task', type=str, default='NER')
parser.add_argument('--tag_scheme', type=str, default='IOBES', choices=['IOB2', 'IOBES'])
parser.add_argument('--train_path', type=str)
parser.add_argument('--dev_path', type=str)
parser.add_argument('--test_path', type=str)
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
parser.add_argument('--learning_alghoritm', type=str, choices=['Adadelta', 'Adagrad', 'SGD'], default='Adadelta')
parser.add_argument('--learning_rate', type=float, default=0.015)
parser.add_argument('--decay', type=float, default=0.05)
parser.add_argument('--grad_clipping', type=float, default=5.0)
parser.add_argument('--feature_column', type=int, default=2)
parser.add_argument('--features', type=int, default=0)

args = parser.parse_args()
task=args.task
tag_scheme=args.tag_scheme
train_path=args.train_path
dev_path=args.dev_path
test_path=args.test_path
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
embeddings_metadata= {'word_embedding':'metadata_tokens.tsv', 'char_embedding':'metadata_characters.tsv'}
tokensfile  = open(embeddings_metadata['word_embedding'], "w");
tokenswriter = csv.writer(tokensfile, delimiter=',')

charsfile  = open(embeddings_metadata['char_embedding'], "w");
charswriter = csv.writer(charsfile,delimiter=',')

tokenswriter.writerow([index_to_token[0]]);
charswriter.writerow(['PAD']);

for char, index in char_alphabet.iteritems():
    charswriter.writerow([char]);
for word, index in word_alphabet.iteritems():
    index_to_token.append(word)
    tokenswriter.writerow([word]);
for word, index in label_alphabet.iteritems():
    index_to_label.append(word)


tokensfile.close()
charsfile.close()
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


logger = open('fmeasure.log','w')
if features:
    feature_size=feature_alphabet.size()
else:
    feature_size=0
model = Model(features=features, feature_dim=40, embed_char=embed_char,grad_clipping=grad_clipping,char_dim=char_dim,filters=n_filters,lstm_size=lstm_size,window=window,learning_alghoritm=learning_alghoritm, learning_rate=learning_rate, decay=decay,embedding_weights=embedd_table, dictonary_size=len(index_to_token)-1, MAX_SEQUENCE_LENGTH=X_train.shape[1], MAX_CHARACTER_LENGTH=C_train.shape[2], alfabeth_size=char_embedd_table.shape[0]-1,feature_size=feature_size,tags=len(index_to_label))
plot_model(model.model,'model.png', show_shapes=True);
tensorboard=keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=1, embeddings_layer_names=['word_embedding','char_embedding'], embeddings_metadata=embeddings_metadata)
conlleval= ConllevalCallback(logger,X_te,Y_test,index2word=index_to_token, index2chunk= index_to_label, batch_size=batch, task=task, tag_scheme=tag_scheme)
model.model.fit(X_tr, Y_train, validation_data=(X_de, Y_dev),epochs=epochs, batch_size=batch, callbacks=[conlleval,tensorboard])
logger.close()
