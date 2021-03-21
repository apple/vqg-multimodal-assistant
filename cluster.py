"""
This file performs clustering for diverse beam searcg
"""
import logging
import string
import collections
import sys

sys.path.append("./")
sys.path.append("../")

from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint


logger = logging.getLogger(__name__)


def process_text(text, stem=True):
    """ Tokenize text and stem words removing punctuation """

    text = text.translate(str.maketrans('','', string.punctuation))
    # text = text.translate(str.maketrans('', '', '1234567890'))
    tokens = word_tokenize(text)

    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]

    return tokens


def cluster_texts(texts, clusters):
    """ Transform texts to Tf-Idf coordinates and cluster texts using K-Means """
    stop_words = stopwords.words('english') #+ list(string.punctuation)
    # print('Stop word')
    logger.debug('Initializing tfidf model')
    vectorizer = TfidfVectorizer(tokenizer=process_text,
                                 sublinear_tf=True,
                                 stop_words=None,
                                 lowercase=True)
    logger.debug('Performing tfidf fit transform')
    tfidf_model = vectorizer.fit_transform(texts)
    logger.debug('Performing kmeans')
    km_model = KMeans(n_clusters=clusters)
    logger.debug('Performing kmeans fit')
    km_model.fit(tfidf_model)
    clustering = collections.defaultdict(list)

    for idx, label in enumerate(km_model.labels_):
        clustering[label].append(idx)

    return clustering


if __name__ == "__main__":
    articles = ['This is a sentence', 'This is also a sentence', 'How are you?', 'Where are we']
    clusters = cluster_texts(articles, 2)
    pprint(dict(clusters))