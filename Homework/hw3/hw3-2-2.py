# CSE 547 Homework 3
# Problem 2.2: SDCA

# Brian de Silva
# 1422824

import numpy as np
from mnist import MNIST
import time
import os
import matplotlib.pyplot as plt



# 
# Define some functions and objects that we will need
# 

class SDCA:
	def __init__(self,features,sigma,reg,nClasses,W=None,batchSize=1):
		self.V = features
		self.sigma = sigma
		self.reg = reg
		self.nClasses = nClasses
		self.b = batchSize

		if W = None:
			self.W = np.zeros((self.nClasses,self.V.shape[0]))
		else:
			self.W = W
		self.alpha = None


	# ---Fourier Kernel---
	# 
	# Returns the feature vector of Fourier features
	def FRBFVec(self,x):
		return np.sin(self.V.dot(x) / self.sigma)

	# ---0/1, square, and G-loss---
	# 
	# Gets the 0/1 and square losses simultaneously so we don't have to recompute feature vectors
	def getLoss(self,Y,X):
		model_out = np.empty(Y.shape[0])
		total = 0.
		labels = np.arange(10)
		for k in xrange(Y.shape[0]):
			rbf = self.FRBFVec(X[k,:],self.V,self.sigma)
			model_out[k] = np.argmax(self.W.dot(rbf))
			total += np.sum(((labels==Y[k]).astype(int) - self.W.dot(rbf)) ** 2)
		z1_out = (1.0 * np.count_nonzero(model_out - Y)) / Y.shape[0]
		sq_out = (total + self.reg * np.sum(self.W**2) / 2.) / Y.shape[0]
		return (sq_out,z1_out)

	# ---G-loss---
	# 
	# Returns the current G-loss
	# ***Should change if alpha changed to have nClasses rows instead of nClasses columns***
	def gLoss(self,Y):
		loss = (np.sum(self.alpha**2) + self.reg * np.sum(self.W**2)) / 2.0
		for k in np.arange(self.nClasses):
			loss -= self.alpha[:,k].dot((Y==k).astype(int))
		return loss

	
	# ---Train---
	# 
	# Trains the model (linear regression) for self.nEpochs passes over the data
	# 
	# 	-XTrain contains the data points for the training set (as rows)
	# 	-XTest contains the data points for the test set (as rows)
	# 	-YTrain contains the labels for the training set
	# 	-YTest contains the labels for the test set
	# 	-nEpochs is the number of epochs that should be run

	def train(self,XTrain,XTest,YTrain,YTest,nEpochs=10):
		tTrain0 = time.time()			# Start a timer

		N = XTrain.shape[0]				# Number of samples
		# step0 = 1.e-5 / 2.				# Initial step size
		# step = step0

		# Vectors to store the loss
		loss = {}
		loss['squareTrain'] = np.zeros(numEpochs + 1)
		loss['squareTest'] = np.zeros(numEpochs + 1)
		loss['z1Train'] = np.zeros(numEpochs + 1)
		loss['z1Train'] = np.zeros(numEpochs + 1)
		loss['gTrain'] = np.zeros(numEpochs + 1)

		# Loop over epochs
		for epoch in xrange(nEpochs):
			tEpoch0 = time.time()

			# Check the square, 0/1, and G-loss once per epoch
			(loss['squareTrain'][epoch], loss['z1Train'][epoch]) = self.getLoss(YTrain,XTrain)
			(loss['squareTest'][epoch], loss['z1Test'][epoch]) = self.getLoss(YTest,XTest)
			loss['gTrain'] = self.gLoss(YTrain)
			tLoss = time.time()

			print "Time to compute the loss: %f" %(tLoss - tEpoch0)
			print "Square loss on training set after %d epochs: %f" %(epoch, loss['squareTrain'][epoch])
			print "0/1 loss on training set after %d epochs: %f" %(epoch, loss['z1Train'][epoch])
			print "G-loss on training set after %d epochs: %f" %(epoch, loss['gTrain'][epoch])

			# Compute new order in which to visit indices
			sampleIndices = np.random.permutation(N)

			# Loop over all the points, updating different coordinates of alpha each time
			for it in xrange(N / self.b):
				currentIndices = sampleIndices[it * self.b : (it+1) * self.b]
				
















# -------------------------------------------------------------------------------------

# 
# Code to run everything
# 


# Import the data
mndata = MNIST('/home/brian/Documents/Coursework/Fall2016/CSE546/hw/hw1/mnist')
train_img, train_label = mndata.load_training()
test_img, test_label = mndata.load_testing()

# Convert to numpy arrays
train_img = np.array(train_img)
train_label = np.array(train_label,dtype=int)
test_img = np.array(test_img)
test_label = np.array(test_label,dtype=int)


# Compute SVD
if os.path.isfile("V.npy"):
	V = np.load("V.npy")
else:
	V = np.linalg.svd(train_img,False)[2]
	np.save("V",V)

# **Note** This V is actually V.H in the traditional SVD: X = U*S*V.H

# Project onto the first 50 singular vectors
trainProj = train_img.dot(V[:50,:].T)
testProj = test_img.dot(V[:50,:].T)
# Note: to get true projection, do trainProj.dot(V[:50,:])

# Estimate a good value of sigma for RBFVec using the median trick
N = trainProj.shape[0]
numSamples = 100
dists = 0.
inds = np.empty(2)
for k in xrange(0,numSamples):
	inds = np.random.choice(np.arange(0,N),2,replace=False)	# Get a random pair of data points
	dists += np.sqrt(np.sum((trainProj[inds[0],:] - trainProj[inds[1],:])**2))

# Cheat a little and use the empirical mean distance between points
dists /= numSamples

# Generate or load random vectors for features
num_features = 30000
if os.path.isfile("feats.npy"):
	feats = np.load("feats.npy")
else:
	feats = np.random.randn(num_features,trainProj.shape[1])
	np.save("feats",feats)


# Set parameters
# --------------------------------------------------------------------------
sigma = dists / 2.
nClasses = 10		# Number of classes
reg = 0.			# Regularization parameter
batchSize = 1		# Batch size for SGD
numEpochs = 10		# Number of epochs to go through before terminating
# --------------------------------------------------------------------------

















# Import the data
mndata = MNIST('/home/brian/Documents/Coursework/Fall2016/CSE546/hw/hw1/mnist')
train_img, train_label = mndata.load_training()
test_img, test_label = mndata.load_testing()

# Convert to numpy arrays
train_img = np.array(train_img)
train_label = np.array(train_label,dtype=int)
test_img = np.array(test_img)
test_label = np.array(test_label,dtype=int)