# Author: Yerbol Aussat

import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from math import log
import operator

# loading training data
def loadTrain():
	global trn_labels
	global trn_data
	trn_labels = []
	
	# total number of words in training examples
	num_words = sum(1 for line in open('words.txt'))
	
	# total number of document (training examples)
	f_labels = open('trainLabel.txt')
	for label in f_labels:
		trn_labels.append(int(label))
	num_docs = len(trn_labels)
	
	global trn_data
	trn_data = np.zeros( (num_docs, num_words) )
	#print "Shape of the train matrix", trn_data.shape
	
	f=open('trainData.txt')
	for line in f:
		doc, word = [int(x) for x in line.split()]
		trn_data[doc - 1, word - 1] = 1
				
# loading testing data
def loadTest():
	global test_labels
	global test_data

	test_labels = []	
	num_words = sum(1 for line in open('words.txt'))
	
	f_labels = open('testLabel.txt')
	for label in f_labels:
		test_labels.append(int(label))
	num_docs = len(test_labels)
	
	global test_data
	test_data = np.zeros( (num_docs, num_words) )
	#print "Shape of test matrix", test_data.shape
	
	f=open('testData.txt')
	for line in f:
		doc, word = [int(x) for x in line.split()]
		test_data[doc - 1, word - 1] = 1

# Loading words from the file	
def loadWords():
	global wordsDictionary
	wordsDictionary = {}
	wordID = 0
	f=open('words.txt')
	for line in f:
		wordsDictionary[wordID] = line[:-1]
		wordID = wordID + 1

def NBLearn(feature_matrix, labels):
	# available labels:
	global ls
	ls = [1, 2]
	num_words = feature_matrix.shape[1]
	
	# Out hypothesis is {thetas and theta0}
	global thetas
	global theta0
	thetas = np.zeros((num_words, len(ls)))
	theta0 = 1.0* (labels.count(1) + 1) / (len(labels) + len(ls))
	
	# populating thetas
	for l in ls:
		feature_matrix_for_l = getFeatureMatrixForLabel(l, feature_matrix, labels)
		for word in range(num_words):	
			array_for_word = feature_matrix_for_l[:, word]
			number_occurences = 1.0 * np.count_nonzero(array_for_word == 1)
			#print "label", l, "number of occurences", number_occurences
			# calculating thetas, using Laplace smoothing:
			thetas[word, l-1] = (number_occurences + 1) / (len(array_for_word) + 2)

# get feature matrix for a specific label
def getFeatureMatrixForLabel(l, feature_matrix, labels):
	feature_matrix_for_l_asList = [feature_matrix[S] for S in range(len(feature_matrix)) if labels[S] == l]
	feature_matrix_for_l = np.array(feature_matrix_for_l_asList)
	return feature_matrix_for_l
		
# predicting class based on the features
def predictBN(feature_vector):
	class_probability = []
	for l in ls:
		prob = probabilityForLabel(feature_vector, l)
		class_probability.append(prob)
	# return class with highest probability
	return ls[class_probability.index(max(class_probability))]

# determine probability for a label
def probabilityForLabel(feature_vector, l):
	if l == ls[0]:
		probability = log(theta0)
	elif l == ls[1]:
		probability = log(1 - theta0)
	
	for wordID in range(len(feature_vector)):
		if feature_vector[wordID] == 1:
			probability += log(thetas[wordID, l-1])
		elif feature_vector[wordID] == 0:
			probability += log(1 - thetas[wordID, l-1])
	return probability

# predict labels for a list of documents:
def predictManyDocs(test_data):
	predictionsList = []
	for i in range(len(test_data)):
		prediction = predictBN(test_data[i])
		#print "Actual label:", test_labels[i], "Predicted:", prediction
		predictionsList.append(prediction)
	return predictionsList

# finding most discriminative word features
def findDiscrFeatures():
	differences = {}
	for wordID in range(len(thetas)):
		dif = abs(log(thetas[wordID, 0]) - log(thetas[wordID, 1]))
		differences[wordID] = dif
	
	sorted_diff = sorted(differences.items(), key=operator.itemgetter(1), reverse = True)
	
	for i in range(10):
		wordID, value = sorted_diff[i]
		print wordID,
		print "(" + wordsDictionary[wordID] + ")"

def main():
	print "Loading data from files ......"
	loadTrain()
	loadTest()
	loadWords()
	
	print "Learning Decision Tree......."
	NBLearn(trn_data, trn_labels)
	
	print "-----------------------------------------"
	print "Prediction:"
	test_pred = predictManyDocs(test_data)
	print "Testing Data Accuracy: ", accuracy_score(test_labels, test_pred) * 100	
	trn_pred = predictManyDocs(trn_data)
	print "Training Data Accuracy: ", accuracy_score(trn_labels, trn_pred) * 100	

	print "-----------------------------------------"
	print "\nPrediction with scikit-learn"
	clf = BernoulliNB(alpha=1, binarize=None)
	clf.fit(trn_data, trn_labels)
	test_pred = clf.predict(test_data)
	print "Testing Data Accuracy: ", accuracy_score(test_labels, test_pred) * 100	
	trn_pred = clf.predict(trn_data)
	print "Training Data Accuracy: ", accuracy_score(trn_labels, trn_pred) * 100	
	
	print "-----------------------------------------"
	print "\nMost Discriminative Word Features:"
	findDiscrFeatures()
	
if __name__ == "__main__":
    main()

