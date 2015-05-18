

import numpy as np
import matplotlib.pyplot as plt
from featureExtractor import FeatureExtractor
from regression import RegressionModel
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn import tree

import random


class NonLinearModels:


	def loadData(self):
		'''
		initialises the data and loads it from hdfs
		and prepares the data.

		'''

		self.training_data, self.test_data = RegressionModel.load_datasets(balance= True, viral_threshold=50000)

		self.viral_threshold = 50000

		#  picking a subset of training data as training with full data takes a lot of time.

		self.X_train = self.training_data[:30000, :-1]
		self.Y_train = self.training_data[:30000,-1]
		self.X_test = self.test_data[:, :-1]
		self.Y_test = self.test_data[:, -1]
		self.test_median = np.median(self.Y_test)
		self.train_median = np.median(self.Y_train)

		self.train_median = self.viral_threshold
		self.test_median = self.viral_threshold
		print "Test Median"
		print self.test_median
		print "Training Median"
		print self.train_median
		# exit()
		


	def trainSVR(self, ctype =1):
		'''
		trains a Support Vector Regression Model from scikit 

		'''
		self.svr_rbf = SVR(kernel='rbf', C = 10, gamma = 0.1)
		self.svr_rbf.fit(self.X_train, self.Y_train)


	def testSVR(self):
		'''
		tests the trained SVR model and calcualtes the rms 
		'''
		self.X_test = self.test_data[:, :-1]
		self.Y_test = self.test_data[:, -1]
		preds = self.svr_rbf.predict(self.X_test)
		rms = np.mean((preds - self.Y_test) ** 2) 
		print rms


	def trainSVC(self):
		'''
		trains a Support Vector Classifier 

		'''
		Y = np.zeros_like(self.Y_train)
		Y[self.Y_train >  self.train_median] = 1
		self.SVC = SVC(kernel = 'rbf', C = 1.0, gamma = 0.01)
		self.SVC.fit(self.X_train, Y)



	def trainDT(self):
		'''
		trains a decision tree classifier 
		'''
		self.DT = tree.DecisionTreeClassifier()
		medianVal = np.median(self.Y_train)

		self.DT = self.DT.fit(self.X_train, self.Y_train)


	def scoreDT(self):
		'''
		evaluates the trained DT.
		'''
		test_median = np.median(self.Y_test)

		Y = np.zeros_like(self.Y_test)
		Y[self.Y_test > test_median] = 1
		Ypreds = self.DT.predict(self.X_test)

		tp = np.sum(Y == Ypreds)

		sr = tp/float(len(Y))
		print sr



	def scoreSVC(self):
		# test_median = np.median(self.Y_test)
		print "Test Median"
		print self.test_median

		Y = np.zeros_like(self.Y_test)
		Y[self.Y_test > self.train_median] = 1
		print self.Y_test.shape
		Ypreds = self.SVC.predict(self.X_test)
		print Ypreds.shape
		print Y.shape

		print np.sum(Y ==1)
		print np.sum(Y == 0)

		print np.sum(Y == Ypreds)
		# print np.sum(Y==Ypreds and Ypreds == 1)
		tp = np.sum(np.equal(Y == Ypreds, Ypreds == 1))
		fp = np.sum(np.equal(Y != Ypreds, Ypreds == 1))
		tn = np.sum(np.equal(Y == Ypreds, Ypreds == 0))
		fn = np.sum(np.equal(Y != Ypreds, Ypreds == 0))
		# print tp
		# print fp 
		# print tn
		# print fn
		tpr = tp / float(tp + fn)
		fpr = fp / float(fp +tn)

		sr = np.sum(Y == Ypreds)/float(Y.shape[0])
		print sr



if __name__ == "__main__":
	nlm = NonLinearModels()
	nlm.loadData()
	# nlm.trainSVR()
	# nlm.testSVR()
	nlm.trainSVC()
	nlm.scoreSVC()
	# nlm.trainDT()
	# nlm.scoreDT()

