
from __future__ import print_function

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np

###Scraping news headlines and descriptions from NIST's webpage###
#you can use this guide to scrape data from a webpage: http://docs.python-guide.org/en/latest/scenarios/scrape/

from lxml import html
import requests

page = requests.get('http://www.nist.gov/allnews.cfm?s=01-01-2014&e=12-31-2014')
tree = html.fromstring(page.content)

#list of news headlines and descriptions
headlines = tree.xpath('//div[@class="select_portal_module_wrapper"]/a/strong/text()')
descriptions = tree.xpath('//div[@class="select_portal_module_wrapper"]/p/text()')
news=[]
for headline in headlines:
	for description in descriptions:
		news.append(headline+description)

#Convert a collection of raw documents to a matrix of TF-IDF features
print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
vectorizer = TfidfVectorizer(input=news, max_df=0.5, min_df=2, stop_words='english')
X = vectorizer.fit_transform(news) 

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()

###let's do some clustering###
#number of clusters, which is 15 since NIST has 15 subject areas
k = 15
km = KMeans(n_clusters=k, init='k-means++', max_iter=100)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()

print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()
for i in range(k):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()
