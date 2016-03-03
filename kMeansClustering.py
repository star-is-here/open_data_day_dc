###Scraping news headlines and descriptions from NIST's webpage###
#you can use this guide to scrape other data from a webpage: http://docs.python-guide.org/en/latest/scenarios/scrape/
from __future__ import print_function
from lxml import html
import requests

print("Retrieving data from NIST")

#retrieve the data from the web page
page = requests.get('http://www.nist.gov/allnews.cfm?s=01-01-2014&e=12-31-2014') 
#use html module to parse it out and store in tree
tree = html.fromstring(page.content)

#create list of news headlines and descriptions. This required obtaining the XPath of the elements by examining the web page.
list_of_headlines = tree.xpath('//div[@class="select_portal_module_wrapper"]/a/strong/text()')
list_of_descriptions = tree.xpath('//div[@class="select_portal_module_wrapper"]/p/text()')
#combine each headline and description into one value in a list
news=[]
for each_headline in list_of_headlines:
	for each_description in list_of_descriptions:
		news.append(each_headline+each_description)

print("Last item in list retrieved: %s" % news[-1])

#Convert a collection of raw documents to a matrix of TF-IDF features

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from time import time

print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
#create a sparse word occurrence frequency matrix of the most frequent words
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
km.fit(X) 								#what's happening here??
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
