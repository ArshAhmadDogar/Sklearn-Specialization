#!/usr/bin/env python3

import pprint # pretty printer
import logging
from sklearn.datasets import fetch_20newsgroups
from fda_helper import preprocess_data
from gensim import corpora, models, similarities

import os.path
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt


# enable logging to display what is happening
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# read dataset 20newsgroups
dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data

texts = preprocess_data(documents)
dictionary = corpora.Dictionary(texts)

bow_corpus = [dictionary.doc2bow(text) for text in texts] # bow = Bag Of Words
# pprint.pprint(bow_corpus[5]) # one example document, words maped to ids

tfidf = models.TfidfModel(bow_corpus) # train tf-idf model
corpus_tfidf = tfidf[bow_corpus] # apply transformation on the whole corpus

##  TODO: transform your tfidf model into a LSI Model
##  using python gensim, use num_topics=200
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=300)
corpus_lsi = lsi[corpus_tfidf]

## TODO: query! pick a random document and formulate a query based on the
## terms in the document.

path="Data"
file_name="myarticles.txt"

documents_list = []
titles=[]
with open( os.path.join(path, file_name) ,"r",encoding="utf8") as fin:
    for line in fin.readlines():
        text = line.strip()
        documents_list.append(text)
print("Total Number of Documents:",len(documents_list))
titles.append( text[0:min(len(text),100)] )

texts = preprocess_data(documents_list)

## TODO: initialize a query structure for your LSI space

dictionary = corpora.Dictionary(texts)
# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in texts]


## TODO: perform the query on the LSI space, interpret the result and summarize your findings in the report

number_of_topics=7
words=10

lsamodel = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model
print(lsamodel.print_topics(num_topics=number_of_topics, num_words=words))


