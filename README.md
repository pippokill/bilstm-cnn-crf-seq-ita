Bi-directional LSTM-CNNs-CRF for Italian Sequence Labeling
========

This is a sequence labeler that supports the following tasks:

* Named Enitity Recognition
* Parts of Speech Tagging
* SuperSense Tagging (use the Named Entity Recognition mode)

If you use this software please cite:

*Pierpaolo Basile, Giovanni Semeraro, Pierluigi Cassotti*. Bi-directional LSTM-CNNs-CRF for Italian Sequence Labeling, Fourth Italian Conference on Computational Linguistics (CLIC-it 2017), 2017.

Requirements
---------------

* Python 3.6
* Numpy
* Tensorflow https://www.tensorflow.org/install/
* Theano http://deeplearning.net/software/theano/install.html
* Keras https://github.com/phipleg/keras/tree/crf

How to use it
----------------

Run **main.py**  
Main supports the following parameters:

* **task** Task  type **default='NER' choices=['NER','POS']**
* **tag_scheme** Tag scheme **default='IOBES' choices=['IOB2', 'IOBES']**
* **train_path** Training file path
* **dev_path** Development file path
* **test_path** Test file path
* **embed_char** 1 for using character embeddings 0 otherwise **default=1**
* **lowerword** 1 for performing token lowercase **default=1**
* **char_dim** Character embeddings dimension **default=30**
* **word_dim** Word embeddings dimension **default=300**
* **word2vecBINARY** 1 if word2vec embeddings is binary 0 otherwise **default=0**
* **label_column** The column index of labels **default=3**
* **word_column** The column index of words **default=0**
* **embedding** Word embeddings tools **default='word2vec' choices=['word2vec','glove','senna','random']**
* **dataset_encoding** The dataset encoding format **default='utf-8'**
* **embedding_encoding** The embeddings encoding format **default='utf-8'**
* **n_filters** Filter number of convolutional networks **default=30**
* **window_size** Window size of convolutional networks **default=3**
* **lstm_size** The state size of Long Short Term Memory **default=200**
* **epochs** Number of epochs **default=50**
* **batch_size** Batch size **default=10**
* **learning_alghoritm** Optimization method **choices=['Adadelta', 'Adagrad', 'SGD'] default='Adadelta'**
* **learning_rate** The learning rate in case of SGD **default=0.015**
* **decay** The deacy rate in case of SGD **default=0.05**
* **grad_clipping** The clip value  **default=5.0**
* **feature_column** The column index of the feature **default=2**
* **features** 1 for using features 0 otherwise **default=0**
