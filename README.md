# bilstm-cnn-crf-seq-ita
Bi-directional LSTM-CNNs-CRF for Italian Sequence Labeling
========

This is a sequence labeler that supports the following tasks:

* Named Enitity Recognition
* Parts of Speech Tagging
* SuperSense Tagging (use the Named Entity Recognition mode)

If you use this software please cite:

**Pierpaolo Basile, Giovanni Semeraro, Pierluigi Cassotti**. Bi-directional LSTM-CNNs-CRF for Italian Sequence Labeling, Fourth Italian Conference on Computational Linguistics (CLIC-it 2017), 2017.

How to use it
----------------

Run **main.py**  
Main supports the following parameters:

* **task** String that defines the task **default='NER' choices=['NER','POS']**
* **tag_scheme** String that defines the tag scheme **default='IOBES' choices=['IOB2', 'IOBES']**
* **train_path** String that defines the training file path
* **dev_path** String that defines the development file path
* **test_path** String that defines the test file path
* **embed_char** 1 for using character embedding 0 otherwise **default=1**
* **lowerword** 1 for performing token lowercase **default=1**
* **char_dim** Character embedding dimension **default=30**
* **word_dim** Word embedding dimension **default=300**
* **word2vecBINARY** 1 if word2vec embedding is binary 0 otherwise **default=0**
* **label_column** The column index of labels **default=3**
* **word_column** The column index of words **default=0**
* **embedding** Word embedding tools **default='word2vec' choices=['word2vec','glove','senna','random']**
* **dataset_encoding** The dataset encoding format **default='utf-8'**
* **embedding_encoding** The embedding encoding format **default='utf-8'**
* **n_filters** Filter number of convolutional networks **default=30**
* **window_size** Window size of convolutional networks **default=3**
* **lstm_size** The state size of Long Short Term Memory **default=200**
* **epochs** Number of epochs **default=50**
* **batch_size** Batch size **default=10**
* **learning_alghoritm** The string that defines the optimizer method **choices=['Adadelta', 'Adagrad', 'SGD'] default='Adadelta'**
* **learning_rate** The learning rate in case of SGD **default=0.015**
* **decay** The deacy rate in case of SGD **default=0.05**
* **grad_clipping** The clip value  **default=5.0**
* **feature_column** The column index of the feature you want to use **default=2**
* **features** 1 for using features 0 otherwise **default=0**
