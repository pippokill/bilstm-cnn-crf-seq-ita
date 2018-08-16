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

import numpy as np
import gensim
import gzip
import theano
import logging
import sys
from subprocess import Popen, PIPE, STDOUT
from keras.utils.data_utils import get_file
from keras.callbacks import Callback


def tag2index(tag_scheme):
    return {tag: i for i, tag in enumerate(tag_scheme)}

def check_tag_scheme(index_to_label):
    new_tag_scheme=[]
    new_tag_scheme.append('PAD')
    entities=[]
    right_set=[]
    right_set.append('O')
    for tag in index_to_label:
        if tag=='O':
            continue
        if tag=='PAD':
            continue
        t=tag.split('-')[1]
        if t not in entities:
            entities.append(t)
    for e in entities:
        right_set.append('B-'+e)
        right_set.append('I-'+e)
        right_set.append('S-'+e)
        right_set.append('E-'+e)
    for tag in index_to_label:
        if tag not in new_tag_scheme:
            new_tag_scheme.append(tag)
    for tag in right_set:
        if tag not in index_to_label and tag not in new_tag_scheme:
            new_tag_scheme.append(tag)
    return new_tag_scheme


def get_logger(name, level=logging.INFO, handler=sys.stdout,
               formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)
    stream_handler = logging.StreamHandler(handler)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

def load_word_embedding_dict(embedding, embedding_path, word_alphabet, logger, embedding_encoding='utf-8',word2vecBINARY=False, embedd_dim=100):
    """
    load word embeddings from file
    :param embedding:
    :param embedding_path:
    :param logger:
    :return: embedding dict, embedding dimention, caseless
    """
    if embedding == 'word2vec':
        # loading word2vec
        logger.info("Loading word2vec ...")
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(embedding_path, encoding=embedding_encoding, binary=word2vecBINARY)
        embedd_dim = word2vec.vector_size
        return word2vec, embedd_dim, False
    elif embedding == 'glove':
        # loading GloVe
        logger.info("Loading GloVe ...")
        embedd_dim = -1
        embedd_dict = dict()
        with gzip.open(embedding_path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue
                tokens = line.split()
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim], dtype=theano.config.floatX)
                embedd[:] = tokens[1:]
                embedd_dict[tokens[0].decode(embedding_encoding)] = embedd
        return embedd_dict, embedd_dim, True
    elif embedding == 'senna':
        # loading Senna
        logger.info("Loading Senna ...")
        embedd_dim = -1
        embedd_dict = dict()
        with gzip.open(embedding_path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split()
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim], dtype=theano.config.floatX)
                embedd[:] = tokens[1:]
                embedd_dict[tokens[0].decode(embedding_encoding)] = embedd
        return embedd_dict, embedd_dim, True
    elif embedding == 'random':
        # loading random embedding table
        logger.info("Loading Random ...")
        embedd_dict = dict()
        words = word_alphabet.get_content()
        scale = np.sqrt(3.0 / embedd_dim)
        for word in words:
            embedd_dict[word] = np.random.uniform(-scale, scale, [1, embedd_dim])
        return embedd_dict, embedd_dim, False
    else:
        raise ValueError("embedding should choose from [word2vec, senna, random, glove]")


def iob2(tags,index_to_label,label_to_index):
    for i, tag in enumerate(tags):
        if index_to_label[tag] == 'PAD':
            continue
        if index_to_label[tag] == 'O':
            continue
        split = index_to_label[tag].split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or index_to_label[tags[i - 1]] == 'O':  # conversion IOB1 to IOB2
            tags[i] = label_to_index['B' + index_to_label[tag][1:]]
        elif index_to_label[tags[i - 1]][1:] == index_to_label[tag][1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = label_to_index['B' + index_to_label[tag][1:]]
    return tags

def iobes_iob(tags):
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'PAD':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags

def iob_iobes(tags,index_to_label,label_to_index):
    new_tags = []
    for i, tag in enumerate(tags):
        if index_to_label[tag] == 'PAD':
            new_tags.append(tag)
        elif index_to_label[tag] == 'O':
            new_tags.append(tag)
        elif index_to_label[tag].split('-')[0] == 'B':
            if i + 1 != len(tags) and \
                index_to_label[tags[i + 1]].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(label_to_index[index_to_label[tag].replace('B-', 'S-')])
        elif index_to_label[tag].split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    index_to_label[tags[i + 1]].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(label_to_index[index_to_label[tag].replace('I-', 'E-')])
        else:
            print(tag)
            raise Exception('Invalid IOB format!')
    return new_tags

class ConllevalCallback(Callback):
    '''Callback for running the conlleval script on the test dataset after
    each epoch.
    '''
    def __init__(self, logger,X_test, y_test, index2word=None, index2chunk=None, batch_size=1, task='NER', tag_scheme='IOBES'):
        if len(X_test)==3:
            self.X_words_test, self.X_char_test, self.X_feature_test= X_test
            self.char=True
            self.features=True
        elif len(X_test)==2:
            self.X_words_test, self.X_char_test= X_test
            self.char=True
            self.features=False
        else:
            self.X_words_test= X_test
            self.char=False
            self.features=False
        self.y_test = y_test
        self.batch_size = batch_size
        self.index2word = index2word
        self.index2chunk = index2chunk
        self.logger=logger
        self.task=task
        self.tag_scheme=tag_scheme

    def on_epoch_end(self, epoch, logs={}):
        self.model.save("/models/"+str(epoch)+".h5")
        if self.char:
            X_test = [self.X_words_test, self.X_char_test]
            if self.features:
                X_test.append(self.X_feature_test)
        else:
            if self.features:
                X_test= [self.X_words_test,self.X_feature_test]
            else:
                X_test= self.X_words_test
        pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(pred_proba, axis=2)
        run_conlleval(self.logger,self.X_words_test, self.y_test, y_pred, self.index2word, self.index2chunk, task=self.task, tag_scheme=self.tag_scheme)


def run_conlleval(logger,X_words_test, y_test, y_pred, index2word, index2chunk, pad_id=0, task='NER', tag_scheme='IOBES'):
    '''
    Runs the conlleval script for evaluation the predicted IOB-tags.
    '''
    url = 'http://www.cnts.ua.ac.be/conll2000/chunking/conlleval.txt'
    path = get_file('conlleval',
                    origin=url,
                    md5_hash='61b632189e5a05d5bd26a2e1ec0f4f9e')

    p = Popen(['perl', path], stdout=PIPE, stdin=PIPE, stderr=STDOUT)

    y_true = np.squeeze(y_test, axis=2)
    sequence_lengths = np.argmax(X_words_test == pad_id, axis=1)
    nb_samples = X_words_test.shape[0]
    conlleval_input = []
    i=0
    for k in range(nb_samples):
        sent_len = sequence_lengths[k]
        words = list(map(lambda idx: index2word[idx], X_words_test[k][:sent_len]))
        true_tags = list(map(lambda idx: index2chunk[idx], y_true[k][:sent_len]))
        pred_tags = list(map(lambda idx: index2chunk[idx], y_pred[k][:sent_len]))
        if task=='NER' and tag_scheme=='IOBES':
            true_tags= iobes_iob(true_tags)
            pred_tags= iobes_iob(pred_tags)
        sent = zip(words, true_tags, pred_tags)
        for row in sent:
            conlleval_input.append(' '.join(row))
        conlleval_input.append('')
    print()
    print(i)
    conlleval_stdout = p.communicate(input='\n'.join(conlleval_input).encode())[0]
    logger.write(conlleval_stdout.decode())
    print(conlleval_stdout.decode())
