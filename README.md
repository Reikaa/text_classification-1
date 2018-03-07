# Text Classification

Text categorization is an important task in natural language processing and information 
retrieval. For instance, news articles, emails or blogs are often classified by topics. 
I implemented a decision tree algorithm and a naive Bayes model in Python to learn 
classifiers that can assign a newsgroup topic to any article. 

A training set and test set of articles with their correct newsgroup label were used.
The articles have been pre-processed and converted to the bag of words model. More 
precisely, each article is converted to a vector of binary values such that each entry 
indicates whether the document contains a specific word or not. 

Correctness os implementation was verified by comparing the results to those obtained 
by the sklearn.tree.DecisionTreeClassifier and sklearn.naive bayes.BernoulliNB classifiers
available in the Python scikit-learn library (www.scikit-learn.org) for machine learning.