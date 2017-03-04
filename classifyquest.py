#File Name: classifyquest.py

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.stem import WordNetLemmatizer
import random
from sklearn import linear_model
import gensim
import operator
import numpy as np

def get_data(filename):
	file = open(filename,'r')

	data = []
	for line in file:
		sentence,category=line.strip().split(' ,,, ')
		data.append((sentence,category))
	return data


test_data = get_data('test_data.txt')

# Remove stopwords and obtain features: 
stop_words=set(stopwords.words("english"))	
w2v = gensim.models.KeyedVectors.load_word2vec_format('glove.6B.50d.txt', binary=False)  

def word_vectorization(wvec,question):
	getvec = []
	for each in question:
		addingup = [0] * 50
		words = word_tokenize(each)
		for w in words:
			w = w.lower()
			if w not in stop_words:
				try:
					addingup = map(operator.add, wvec[w], addingup)
				except KeyError:
					continue
		row = [vec for vec in addingup]
		row = np.sum(row)
		if row:
			if getvec:
				np.append(getvec,row)
			else:
				getvec = row
	return getvec

data = [q[0] for q in test_data]

X = [word_vectorization(w2v, [element[0]]) for element in test_data]


def confidence(prediction,actuals):
	success = [np.any(z[0] == z[1]) for z in zip(prediction, actuals)]
	return (success.count(True)/len(success))*100

Y = [cat[1] for cat in test_data]

classifier = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs')
classifier.fit(np.array(X[:800]).reshape(-1, 1), np.array(Y[:800]))

pred_train = [classifier.predict(np.array(X[:800]).reshape(-1, 1))]
pred_test = [classifier.predict(np.array(X).reshape(-1, 1))]

print ("Confidence:")
print ("Train accuracy " + str(confidence(pred_train, Y[:800])))
print ("Test accuracy " + str(confidence(pred_test, Y)))