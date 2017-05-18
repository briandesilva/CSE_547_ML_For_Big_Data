# CSE 547 Homework 3
# Problem 2.2 and 3: SDCA

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

class SDCAClassifier:
	def __init__(self,features,sigma,reg,nClasses,batchSize=1,gamma=None,W=None):
		self.V = features
		self.sigma = sigma
		self.reg = reg
		self.nClasses = nClasses
		self.b = batchSize

		if gamma is None:
			self.gamma = 1. / self.b
		else:
			self.gamma = gamma
		if W is None:
			self.W = np.zeros((self.nClasses,self.V.shape[0]))
		else:
			self.W = W
		self.alpha = None 			# Need to wait for data to define this


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
			rbf = self.FRBFVec(X[k,:])
			model_out[k] = np.argmax(self.W.dot(rbf))
			total += np.sum(((labels==Y[k]).astype(int) - self.W.dot(rbf)) ** 2)
		z1_out = (1.0 * np.count_nonzero(model_out - Y)) / Y.shape[0]
		sq_out = (total + self.reg * np.sum(self.W**2) / 2.) / Y.shape[0]
		return (sq_out,z1_out)

	# ---G-loss---
	# 
	# Returns the current G-loss
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

		N = XTrain.shape[0]					# Number of samples
		labels = np.arange(self.nClasses)	# Array of classes
		self.alpha = np.zeros((N,nClasses))

		# Vectors to store the loss
		loss = {}
		loss['squareTrain'] = np.zeros(nEpochs + 1)
		loss['squareTest'] = np.zeros(nEpochs + 1)
		loss['z1Train'] = np.zeros(nEpochs + 1)
		loss['z1Test'] = np.zeros(nEpochs + 1)
		loss['gTrain'] = np.zeros(nEpochs + 1)

		# Loop over epochs
		for epoch in xrange(nEpochs):
			tEpoch0 = time.time()

			# Check the square, 0/1, and G-loss once per epoch
			(loss['squareTrain'][epoch], loss['z1Train'][epoch]) = self.getLoss(YTrain,XTrain)
			(loss['squareTest'][epoch], loss['z1Test'][epoch]) = self.getLoss(YTest,XTest)
			if self.b==1:
				loss['gTrain'][epoch] = self.gLoss(YTrain)
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

				# Do one batch of updates

				# Get dAlpha's and update alpha's and w
				WOld = self.W
				for ind in xrange(self.b):
					i = currentIndices[ind]
					xi = self.FRBFVec(XTrain[i,:])
					dAlpha = self.gamma * ((labels==YTrain[i]).astype(int) - WOld.dot(xi) - self.alpha[i,:]) /(1.0 + np.sum(xi**2) / self.reg)
					self.alpha[i,:] += dAlpha
					self.W += np.kron(dAlpha[None].T / self.reg , xi[None])

			# Print time to complete the epoch
			tEpoch1 = time.time()
			print "Time elapsed during epoch %d: %f\n" %(epoch, tEpoch1 - tEpoch0)

		#  If done, compute final losses
		(loss['squareTrain'][-1], loss['z1Train'][-1]) = self.getLoss(YTrain,XTrain)
		(loss['squareTest'][-1], loss['z1Test'][-1]) = self.getLoss(YTest,XTest)
		if self.b==1:
			loss['gTrain'][-1] = self.gLoss(YTrain)

		# Print total time to solve problem
		tTrain1 = time.time()
		print "Time to carry out %d epochs of SDCA with batch size %d: %f\n" %(nEpochs,self.b,tTrain1-tTrain0)

		return loss



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

# Project onto the first 50 singular vectors (overwriting original data to save space)
train_img = train_img.dot(V[:50,:].T)
test_img = test_img.dot(V[:50,:].T)
# Note: to get true projection, do trainProj.dot(V[:50,:])

# Estimate a good value of sigma for RBFVec using the median trick
N = train_img.shape[0]
numSamples = 100
dists = 0.
inds = np.empty(2)
for k in xrange(0,numSamples):
	inds = np.random.choice(np.arange(0,N),2,replace=False)	# Get a random pair of data points
	dists += np.sqrt(np.sum((train_img[inds[0],:] - train_img[inds[1],:])**2))

# Cheat a little and use the empirical mean distance between points
dists /= numSamples

# Generate or load random vectors for features
num_features = 30000
if os.path.isfile("feats.npy"):
	feats = np.load("feats.npy")
else:
	feats = np.random.randn(num_features,train_img.shape[1])
	np.save("feats",feats)


# Set parameters
# --------------------------------------------------------------------------
sigma = dists / 2.
nClasses = 10		# Number of classes
reg = 1.0			# Regularization parameter
batchSize = 1		# Batch size for SDCA
nEpochs = 15			# Number of epochs to go through before terminating
# --------------------------------------------------------------------------
# regs = np.logspace(-5,1,7)
# for reg in regs:

# Initialize model
SDCA = SDCAClassifier(feats,sigma,reg,nClasses,batchSize)

# Train model
loss = SDCA.train(train_img,test_img,train_label,test_label,nEpochs)

# Plot G-loss on training set
epochs = range(nEpochs+1)
plt.figure(1)
plt.plot(epochs,loss['gTrain'],'b-o',linewidth=1.5)
plt.xlabel('Epoch')
plt.ylabel('G-loss')
plt.title('G-loss (SDCA with b=1)')

# Plot square loss on training and test sets
plt.figure(2)
plt.plot(epochs,loss['squareTrain'],'b-o',epochs,loss['squareTest'],'r-o')
plt.xlabel('Epoch')
plt.ylabel('Square loss')
plt.title('Square loss (SDCA with b=1)')
plt.legend(['Training','Test'])

# Plot 0/1 loss on training and test sets
plt.figure(3)
plt.plot(epochs[1:],loss['z1Train'][1:],'b-o',epochs[1:],loss['z1Test'][1:],'r-o')
plt.xlabel('Epoch')
plt.ylabel('0/1 loss')
plt.title('0/1 loss (SDCA with b=1)')
plt.legend(['Training','Test'])

# Zoomed in version of the square loss (omits initial loss)
plt.figure(4)
plt.plot(epochs[1:],loss['squareTrain'][1:],'b-o',epochs[1:],loss['squareTest'][1:],'r-o')
plt.xlabel('Epoch')
plt.ylabel('Square loss')
plt.title('Square loss (SDCA with b=1)')
plt.legend(['Training','Test'])

# Output final loss
print "Square loss (training): %f" %loss['squareTrain'][-1]
print "0/1 loss (training): %f" %loss['z1Train'][-1]
print "Square loss (test): %f" %loss['squareTest'][-1]
print "0/1 loss (test): %f" %loss['z1Test'][-1]
print "G-loss (training): %f" %loss['gTrain'][-1]

# Output total mistakes
print "Total mistakes on training set: %f" %(loss['z1Train'][-1] * train_label.shape[0])
print "Total mistakes on test set: %f" %(loss['z1Test'][-1] * test_label.shape[0])
print "\n\n\n"

# Output: 
# Square loss (training)        : 0.043270
# 0/1 loss (training)           : 0.001333
# Square loss (test)            : 0.090068
# 0/1 loss (test)               : 0.013700
# G-loss (training)             : -2.124226
# Total mistakes on training set: 80.000000
# Total mistakes on test set    : 137.000000






# ------------------------------------------------------------------------------------------
# --------------------------Run again with batchSize = 100----------------------------------
# ------------------------------------------------------------------------------------------
batchSize = 100
print "--------------------------------------------------------------------------------------"
print "Running with batch size = %d" %batchSize
print "--------------------------------------------------------------------------------------"


# Initialize model
SDCA_batch = SDCAClassifier(feats,sigma,reg,nClasses,batchSize)

# Train model
loss_batch = SDCA_batch.train(train_img,test_img,train_label,test_label,nEpochs)

# Plot square loss on training and test sets
epochs = np.arange(nEpochs+1) * train_img.shape[0]
plt.figure(5)
plt.plot(epochs,loss_batch['squareTrain'],'b-o',epochs,loss_batch['squareTest'],'r-o',epochs,loss['squareTrain'],'k--o',epochs,loss['squareTest'],'g--o')
plt.xlabel('Points used')
plt.ylabel('Square loss')
plt.title('Square loss (SDCA with b=100)')
plt.legend(['Training (b=100)','Test (b=100)','Training (b=1)','Test (b=1)'])

# Plot 0/1 loss on training and test sets
plt.figure(6)
plt.plot(epochs[1:],loss_batch['z1Train'][1:],'b-o',epochs[1:],loss_batch['z1Test'][1:],'r-o',epochs[1:],loss['z1Train'][1:],'k--o',epochs[1:],loss['z1Test'][1:],'g--o')
plt.xlabel('Points used')
plt.ylabel('0/1 loss')
plt.title('0/1 loss (SDCA with b=100)')
plt.legend(['Training (b=100)','Test (b=100)','Training (b=1)','Test (b=1)'])

# Plot zoomed-in version of square loss
plt.figure(7)
plt.plot(epochs[1:],loss_batch['squareTrain'][1:],'b-o',epochs[1:],loss_batch['squareTest'][1:],'r-o',epochs[1:],loss['squareTrain'][1:],'k--o',epochs[1:],loss['squareTest'][1:],'g--o')
plt.xlabel('Points used')
plt.ylabel('Square loss')
plt.title('Square loss (SDCA with b=100)')
plt.legend(['Training (b=100)','Test (b=100)','Training (b=1)','Test (b=1)'])
plt.show()

# Print the final losses
print "Square loss (training): %f" %loss_batch['squareTrain'][-1]
print "0/1 loss (training): %f" %loss_batch['z1Train'][-1]
print "Square loss (test): %f" %loss_batch['squareTest'][-1]
print "0/1 loss (test): %f" %loss_batch['z1Test'][-1]
print "G-loss (training): %f" %loss_batch['gTrain'][-1]

# Output total mistakes
print "Total mistakes on training set: %f" %(loss_batch['z1Train'][-1] * train_label.shape[0])
print "Total mistakes on test set: %f" %(loss_batch['z1Test'][-1] * test_label.shape[0])
print "\n\n\n"

# Output:
# Square loss (training)        : 0.087129
# 0/1 loss (training)           : 0.021067
# Square loss (test)            : 0.092429
# 0/1 loss (test)               : 0.023800
# G-loss (training)             : -0.074742
# Total mistakes on training set: 1264.000000
# Total mistakes on test set    : 238.000000




# ------------------------------------------------------------------------------------------
# -------------Run again with batchSize = 100 and different learning rate-------------------
# ------------------------------------------------------------------------------------------

# 
# Empirically determine a good value of gamma
# 

batchSize = 100
gammas = np.linspace(1./batchSize,1,11)[1:]
gammas = [1.5, 2, 4, 10, 20, 40, 100]

for gamma in gammas:
	print "-----------------------------------------------------------------"
	print "-----------------------Testing gamma = %f------------------"%gamma
	print "-----------------------------------------------------------------"

	SDCA_gamma = SDCAClassifier(feats,sigma,reg,nClasses,batchSize,gamma=gamma)

	# Train model
	loss_gamma = SDCA_gamma.train(train_img,test_img,train_label,test_label,nEpochs=1)

	# Print loss
	
	print "\n\nSquare loss (training): %f" %loss_gamma['squareTrain'][-1]
	print "0/1 loss (training): %f" %loss_gamma['z1Train'][-1]
	print "Square loss (test): %f" %loss_gamma['squareTest'][-1]
	print "0/1 loss (test): %f" %loss_gamma['z1Test'][-1]
	print "\n\n"



# 
# Actually run SDCA with best choice of gamma
# 

gamma = 0.85
batchSize = 100

# Initialize model
SDCA_gamma = SDCAClassifier(feats,sigma,reg,nClasses,batchSize,gamma=gamma)

# Train model
loss_gamma = SDCA_gamma.train(train_img,test_img,train_label,test_label,nEpochs)

# Plot square loss on training and test sets
epochs = np.arange(nEpochs+1) * train_img.shape[0]
plt.figure()
plt.plot(epochs,loss_gamma['squareTrain'],'b-o',epochs,loss_gamma['squareTest'],'r-o')
plt.xlabel('Points used')
plt.ylabel('Square loss')
plt.title('Square loss (SDCA with b=100, gamma=0.85)')
plt.legend(['Training','Test'])

# Plot 0/1 loss on training and test sets
plt.figure()
plt.plot(epochs[1:],loss_gamma['z1Train'][1:],'b-o',epochs[1:],loss_gamma['z1Test'][1:],'r-o')
plt.xlabel('Points used')
plt.ylabel('0/1 loss')
plt.title('0/1 loss (SDCA with b=100, gamma=0.85)')
plt.legend(['Training','Test'])

# Zoomed in version of the square loss (omits initial loss)
plt.figure()
plt.plot(epochs[1:],loss_gamma['squareTrain'][1:],'b-o',epochs[1:],loss_gamma['squareTest'][1:],'r-o')
plt.xlabel('Points used')
plt.ylabel('Square loss')
plt.title('Square loss (SDCA with b=100, gamma=0.85)')
plt.legend(['Training','Test'])
plt.show()

# Output final loss
print "Square loss (training): %f" %loss_gamma['squareTrain'][-1]
print "0/1 loss (training): %f" %loss_gamma['z1Train'][-1]
print "Square loss (test): %f" %loss_gamma['squareTest'][-1]
print "0/1 loss (test): %f" %loss_gamma['z1Test'][-1]
print "G-loss (training): %f" %loss_gamma['gTrain'][-1]

# Output total mistakes
print "Total mistakes on training set: %f" %(loss_gamma['z1Train'][-1] * train_label.shape[0])
print "Total mistakes on test set: %f" %(loss_gamma['z1Test'][-1] * test_label.shape[0])
print "\n\n\n"

# Output:
# Square loss (training)        : 0.044858
# 0/1 loss (training)           : 0.001400
# Square loss (test)            : 0.087752
# 0/1 loss (test)               : 0.013800
# G-loss (training)             : 0.000000
# Total mistakes on training set: 84.000000
# Total mistakes on test set    : 138.000000
