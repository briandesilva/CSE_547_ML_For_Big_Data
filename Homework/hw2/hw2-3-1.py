# CSE 547 Homework 2
# Problem 3 - K-means and EM
# Brian de Silva

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#----------------------------------------------------------------------------------------
# 									Function definitions
#----------------------------------------------------------------------------------------


# ---K-means---
# 
# Naive K-means implementation
# 	Input:
# 		-numClasses is the number of clusters
# 		-X is the data matrix (rows are data points)
# 		-MAX_ITS is the maximum number of iterations taken before k-means terminates
# 	Output:
# 		-Y is the set of labels for points in X (labels take values 0,1,...,numClasses-1)
# 		-Mu is the set of means of the classes (ceners[i] corresponds to class i)
# 		-reconErr2 is the square reconstruction error at each iteration
def kmeans(numClasses,X,MAX_ITS=300):
	N = X.shape[0]						# Number of data points
	Y = np.zeros(N).astype(int)			# Labels
	Yprev = np.zeros(N).astype(int)		# Labels from previous iteration
	reconErr2 = np.zeros(MAX_ITS)		# Square reconstruction error

	# TODO: Make sure none of these are too close together
	# Initialize centers as numClasses random points in data
	Mu = X[np.random.choice(np.arange(N),numClasses,replace=False),:]

	# Store square 2-norms of data points to speed up computation
	X2 = np.sum(X**2,1)

	# Iteratively update centers until convergence
	for it in range(MAX_ITS):

		# Compute quantities we can reuse
		Mu2 = np.sum(Mu**2,1)

		# Assign labels and get error (starts at iteration 0)
		for k in range(N):
			dists = Mu2 + X2[k] - 2. * Mu.dot(X[k,:])
			Y[k] = np.argmin(dists)
			reconErr2[it] += dists[Y[k]]

		# Compute new centers
		for k in range(numClasses):
			Mu[k,:] = np.mean(X[Y==k,:],0)

		# Check if we should exit
		if np.array_equal(Y,Yprev):
			break
		else:
			Yprev = np.copy(Y)

	print "%d iterations of k-means to achieve convergence." %(it+1)

	return (Y,Mu,reconErr2[:(it+1)])


# --- Make nice plots of clusters ---
# 
# 	Input:
# 		-numClasses: number of classes (the k in k-means)
# 		-X: the data (points should be rows)
# 		-labels: the labels for the rows in X
def plot_clusters(numClasses,X,labels):
	# TODO: write this function
	for k in xrange(numClasses):
		plt.plot(X[labels==k,0],X[labels==k,1],'o')

	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.title('K = ' + str(numClasses))
	plt.savefig('figures/kmeans_2-3b_k' + str(numClasses) + '.png')
	# plt.show()



#----------------------------------------------------------------------------------------
# 									Numerical tests
#----------------------------------------------------------------------------------------

# Read in the data
data = pd.read_csv('2DGaussianMixture.csv').as_matrix()
true_labels = data[:,0]
X = data[:,1:]

# Apply k-means
ks = [2,3,5,10,15,20]

for k in ks:
	labels, mu, err = kmeans(k,X)
	plot_clusters(k,X,labels)