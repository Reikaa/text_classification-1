# Author: Yerbol Aussat

import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from math import log
import matplotlib.pyplot as plt

def DLT(feature_matrix, labels, attributes, default, max_depth):
	mode = max(set(labels), key=labels.count)

	if max_depth == 0:
		#print "max_depth is reached"
		return mode
	
	if len(feature_matrix) == 0:
		#print "Examples is empty"
		return mode
	elif (labels.count(labels[0]) == len(labels)):
		#print "All examples have the same classification"
		return labels[0]	
	elif len(attributes) == 0:
		#print "No attributes left"
		return mode # MODE
	else:
		best, IG = choose_attribute(feature_matrix, labels, attributes)
		#print "best attribute:", best, "IG", IG

		root = Tree()
		root.attr = best
		root.IG = IG
		for val in range(2):
			matrixForAttrValue, labelsForAttrValue = findMattrixForAttrVal(feature_matrix, labels, best, val)
			new_attr = [attributes[i] for i in range(len(attributes)) if attributes[i] != best]
			
			#print "value: ", val
			subtree = DLT(matrixForAttrValue, labelsForAttrValue, new_attr, mode, max_depth - 1)
			root.subtrees[val] = subtree
		return root
		
# choose the best attribute
def choose_attribute(feature_matrix, labels, attributes):
	max_attr = -1 # attribute with maximum information gain
	max_IG = 0 # maximum information gain
	for attribute in attributes:
		IG = findIG(feature_matrix, labels, attribute)
		if IG > max_IG:
			max_IG = IG
			max_attr = attribute
		#print "IG for ", attribute, " is ", IG
	#print "attribute with max IG", max_attr, max_IG
	return max_attr, max_IG

# Reduced matrix (feature matrix + labels) for a certain value of attribute
def findMattrixForAttrVal(feature_matrix, labels, attribute, value):
	labelsForAttrVal = [labels[S] for S in range(len(labels)) if feature_matrix[S, attribute] == value]
	matrixForAttrVal_asList = [feature_matrix[S] for S in range(len(feature_matrix)) if feature_matrix[S, attribute] == value]
	matrixForAttrVal = np.array(matrixForAttrVal_asList)
	return matrixForAttrVal, labelsForAttrVal
		
# Find labels for the attribute value		
def findLabelsForAttrVal(feature_matrix, labels, attribute, value):
	labelsForAttrVal = [labels[S] for S in range(len(labels)) if feature_matrix[S, attribute] == value]
	return labelsForAttrVal
		
# Find entropy
def findI(p, n):
	if (p==0 or n==0):
		return 0
	return -p/(p+n) * log((p/(p+n)), 2) - n/(p+n) * log((n/(p+n)), 2) 

# find the Information Gain for attribute
def findIG(feature_matrix, labels, attribute):
	p = 1.0*labels.count(1)
	n = 1.0*labels.count(2)
	I = findI(p, n)
	remainder = 0
	for value in range(2):
		labelsForAttrVal = findLabelsForAttrVal(feature_matrix, labels, attribute, value)
		p_val = 1.0*labelsForAttrVal.count(1)
		n_val = 1.0*labelsForAttrVal.count(2)
		
		remainder += (p_val+n_val) / (p+n) * findI(p_val, n_val)
		#print "size of reduced matrix for a value", value, " " , len(matrixForAttrVal)	
	IG = I - remainder
	return IG

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

# printing the tree
def printTree(dtree, space): 
	if not isinstance(dtree, Tree):
		print getClass(dtree)
	else:
		print dtree.attr, "(" + wordsDictionary[dtree.attr] + ") IG:", dtree.IG
		for val in dtree.subtrees:
			print space, getVal(val), 
			printTree(dtree.subtrees[val], space + "   ")
	
# 0 = "word is absent in the document"	
# 1 = "word is present in the document"	
def getVal(v):
	if v == 0:
		return "absent:"
	elif v == 1:
		return "present:"

# 1 = "atheism"
# 2 = "graphics
def getClass(leaf)	:
	if leaf == 1:
		return "class1 (atheism)"
	elif leaf == 2:
		return "class2 (graphics)"
	
	
# predicting class based on the features
def predictDLT(feature_vector, dtree):
	if not isinstance(dtree, Tree):
		#print "prediction:", getClass(dtree)
		return dtree
	else:
		wordID = dtree.attr
		absent_or_present = feature_vector[wordID] # 0 if absent, 1 if present
		return predictDLT(feature_vector, dtree.subtrees[absent_or_present])

# predict labels for a list of documents:
def predictManyDocs(test_data, dtree):
	predictionsList = []
	for i in range(len(test_data)):
		prediction = predictDLT(test_data[i], decisionTree)
		#print "Actual label:", test_labels[i]
		predictionsList.append(prediction)
	return predictionsList
	
def main():
	
	print "Loading data from files ......"
	loadTrain()
	loadTest()
	loadWords()
	
	print "Learning Decision Tree......."
	attributes = range(trn_data.shape[1])	
	global decisionTree
	depth = 18
	decisionTree = DLT(trn_data, trn_labels, attributes, 1, depth)
	
	
	print "\nDecision Tree: "
	print "-----------------------------------------"
	printTree(decisionTree, "")
	print "-----------------------------------------"

	print "Prediction:"
	test_pred = predictManyDocs(test_data, decisionTree)
	print "Testing Data Accuracy: ", accuracy_score(test_labels, test_pred) * 100	
	trn_pred = predictManyDocs(trn_data, decisionTree)
	print "Training Data Accuracy: ", accuracy_score(trn_labels, trn_pred) * 100	
		
	print "-----------------------------------------"
	print "\nPrediction with scikit-learn"
	clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = depth)
	clf.fit(trn_data, trn_labels)
	test_pred = clf.predict(test_data)
	print "Testing Data Accuracy: ", accuracy_score(test_labels, test_pred) * 100
	trn_pred = clf.predict(trn_data)
	print "Training Data Accuracy: ", accuracy_score(trn_labels, trn_pred) * 100	

	'''
	# Graph showing the training and testing accuracy vs max_depth
	d_list = [0, 3, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 70, 100]
	testAccList = []
	trnAccList = []

	print "Learning Decision Tree......."
	attributes = range(trn_data.shape[1])	
	global decisionTree

	for d in d_list:
		decisionTree = DLT(trn_data, trn_labels, attributes, 1, d)
		test_pred = predictManyDocs(test_data, decisionTree)
		trn_pred = predictManyDocs(trn_data, decisionTree)
		testAccuracy = accuracy_score(test_labels, test_pred) * 100	
		trnAccuracy = accuracy_score(trn_labels, trn_pred) * 100	
		testAccList.append(testAccuracy)
		trnAccList.append(trnAccuracy)

	plt.plot(d_list, testAccList, label = "Testing Data Accuracy", color='r')
	plt.plot(d_list, trnAccList, label = "Training Data Accuracy", color='b')
	plt.legend()
	plt.xlabel('Max Depth')
	plt.ylabel('Accuracy (%)')
	plt.title('Training and Testing Accuracy vs Max Depth')
	plt.grid(True)
	plt.savefig("test.png")
	plt.show()
	'''
	
class Tree():
	def __init__(self):
		self.subtrees = {} # subtrees of this tree
		self.attr = None # attribute
		self.IG = None # Information Gain

if __name__ == "__main__":
    main()