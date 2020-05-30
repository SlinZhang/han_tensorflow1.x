# han_tensorflow1.x
Implemented the paper "Hierarchical Attention Networks for Document Classification" by tensorflow
https://www.aclweb.org/anthology/N16-1174.pdf

> the Hierarchical Attention Network
(HAN) that is designed to capture two basic insights
about document structure. First, since documents
have a hierarchical structure (words form sentences,
sentences form a document), we likewise construct a
document representation by first building representations
of sentences and then aggregating those into
a document representation. Second, it is observed
that different words and sentences in a documents
are differentially informative.

# Environment
python 3.6 
tensorflow 1.14.0

# Dataset
You can download the IMDB  dataset from http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

# How to train the model
You can set the hyperparameters of the model in config.py then run the commond
python train.py
 
