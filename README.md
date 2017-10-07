# bilstm-cnn-crf-seq-ita
Bi-directional LSTM-CNNs-CRF for Italian Sequence Labeling
README
========

This is a sequence labeler, supports the following tasks:

* Named Enitity Recognition
* Parts of Speech Tagging
* SuperSense Tagging (For this task, use the Named Entity Recognition mode)

How to use it
-------------

Run **main.py**
Main support the following parameters:

* **task** String that define the task **default='NER' choises=['NER','POS']**
* **tag_scheme** String that define the tag scheme **default='IOBES' choices=['IOB2', 'IOBES']**
* **train_path** String that define the training file's path
* **dev_path** String that define the development file's path
* **test_path** String that define the test file's path
* **embed_char** 1 to use character embedding 0 otherwise **default=1**
* **lowerword** Lowercase of dataset's tokens **default=1**
* **char_dim** Character embedding dimension **default=30**
* **word_dim** Word embedding dimension **default=300**
* **word2vecBINARY** 1 if word2vec embedding is binary 0 otherwise **default=0**
* **label_column** The column index of labels **default=3**
* **word_column** The column index of words **default=0**
* **embedding** Word embedding tools  **default='word2vec' choices=['word2vec','glove','senna','random']**
* **dataset_encoding** The dataset's encoding format **default='utf-8'**
* **embedding_encoding** The embedding's encoding format **default='utf-8'**
* **n_filters** Filter's number of convolutional networks **default=30**
* **window_size** Window's size of convolutional networks **default=3**
* **lstm_size** The state size of Long Short Term Memory **default=200**
* **epochs' Number of epochs **default=50**
* **batch_size** Batch size **default=10**
* **learning_alghoritm** The string that define the optimizer you want to use **choices=['Adadelta', 'Adagrad', 'SGD'] default='Adadelta'**
* **learning_rate** The learning rate if use SGD **default=0.015**
* **decay** The deacy rate if use SGD **default=0.05**
* **grad_clipping** The clip value  **default=5.0**
* **feature_column** The column index of feature you want to use **default=2**
* **features** 1 If want to use features 0 otherwise **default=0**
