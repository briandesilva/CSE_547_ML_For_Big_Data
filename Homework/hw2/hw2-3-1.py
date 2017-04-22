# CSE 547 Homework 2
# Problem 3 - K-means and EM
# Brian de Silva

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

import pdb

#----------------------------------------------------------------------------------------
# 									Function definitions
#----------------------------------------------------------------------------------------


# ---K-means---
# 
# Naive K-means implementation
# 	Input:
# 		-numClasses is the number of clusters
# 		-X is the data matrix (rows are data points)
# 		-Mu is the set of initial centers
# 		-MAX_ITS is the maximum number of iterations taken before k-means terminates
# 	Output:
# 		-Y is the set of labels for points in X (labels take values 0,1,...,numClasses-1)
# 		-Mu is the set of means of the classes (centers[i] corresponds to class i)
# 		-reconErr2 is the square reconstruction error at each iteration
def kmeans(numClasses,X,Mu=None,MAX_ITS=300):
	N = X.shape[0]						# Number of data points
	Y = np.zeros(N).astype(int)			# Labels
	Yprev = np.zeros(N).astype(int)		# Labels from previous iteration
	reconErr2 = np.zeros(MAX_ITS)		# Square reconstruction error

	# Initialize centers as numClasses random points in data
	if Mu is None:
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

	# print "%d iterations of k-means to achieve convergence." %(it+1)

	return (Y,Mu,reconErr2[:(it+1)])


# --- EM for GMM ---
# Expectation maximization Gaussian Mixture Model implementation
# 	Input:
# 		-numClasses is the number of clusters
# 		-X is the data matrix (rows are data points)
# 		-Mu is the set of initial centers
# 		-TOL is the tolerance for how small the log likelihood can change between iterations
# 		-MAX_ITS is the maximum number of iterations taken before the algorithm terminates
# 	Output:
# 		-Y is the set of labels for points in X (labels take values 0,1,...,numClasses-1)
# 		-Mu is the set of means of the classes (centers[i] corresponds to class i)
# 		-S is a list of the covariance matrices
# 		-log_like is the log-likelihood at each iteration
def em_GMM(numClasses,X,Mu=None,TOL=1.e-6,MAX_ITS=300):
	N = X.shape[0]						# Number of data points
	Y = np.zeros(N).astype(int)			# Labels
	log_like = np.zeros(MAX_ITS)		# Square reconstruction error

	# Initialize centers as numClasses random points in data
	if Mu is None:
		Mu = X[np.random.choice(np.arange(N),numClasses,replace=False),:]

	# Initialize covariance matrices
	S = [np.identity(X.shape[1])] * numClasses

	# Initialize the responsibilities
	pi = np.ones(numClasses) / numClasses

	# Matrix for storing likelihoods
	# likelihood_matrix[n,k] = pi_k * N(x_n|mu_k,Sigma_k)
	likelihood_matrix = np.empty((N,numClasses))

	# Matrix for storing responsibilities
	R = np.empty_like(likelihood_matrix)

	for it in xrange(MAX_ITS):

		# Update likelihoods
		for n in xrange(X.shape[0]):
			for k in xrange(numClasses):
				likelihood_matrix[n,k] = pi[k] * multivariate_normal.pdf(X[n,:],Mu[k,:],S[k])
		
		# Store and check log-likelihood
		log_like[it] = np.sum(np.log(np.sum(likelihood_matrix,0)))
		if (it>0) and (np.abs(log_like[it] - log_like[it-1])) < TOL:
			break


		# E step:

		# Precompute denominators
		denom = np.sum(likelihood_matrix,1)

		# Compute responsibilities
		R = likelihood_matrix / denom[:,None]


		# M step:

		# Update means
		Nk = np.sum(R,0)		# Weighted number of points assigned to each class
		Mu = R.T.dot(X) / Nk[:,None]
		# for k in xrange(numClasses):
			# Mu[k,:] = R[:,k].T.dot(X) / Nk[k]

		# Update covariance matrices
		for k in xrange(numClasses):
			# S[k] = ((X - Mu[k,:]).T.dot((X - Mu[k,:]))) / Nk[k]
			XMM = X - Mu[k,:]
			S[k] = np.zeros((X.shape[1],X.shape[1]))
			for n in xrange(N):
				S[k] += R[n,k] * np.outer(XMM[n,:],XMM[n,:])
			S[k] /= Nk[k]
		# Update responsibiliites
		pi = Nk / N

	print "EM for GMM converged after %d iterations." %it
	# Set some output quantities
	Y = np.argmax(R,1)
	return Y, Mu, S, log_like[:(it+1)]




# --- K-means++ intialization algorithm ---
#
# 	Input:
# 		-numClasses: number of classes (the k in k-means)
# 		-X: the data (points should be rows)
def kmeans_pp_init(numClasses,X):
	Mu = np.empty((numClasses,X.shape[1]))

	# Store square 2-norms of data points to speed up computation
	X2 = np.sum(X**2,1)

	# First center is a random point from the data set
	Mu[0,:] = X[np.random.randint(0,X.shape[0]),:]				# Array for centers
	D = np.empty(X.shape[0])											# Array for distances	

	# Get remaining centers
	for j in xrange(1,numClasses):
		# Compute distance from points to previous cluster centers
		Mu2 = np.sum(Mu[:j,:]**2,1)
		for k in xrange(len(D)):
			dists = Mu2 + X2[k] - 2. * Mu[:j,:].dot(X[k,:])
			D[k] = dists.min()

		# Pick next cluster center from X with probability proportional to distance to closest center
		new_cent_id = np.random.choice(range(X.shape[0]),p=(D / np.sum(D)))
		Mu[j,:] = X[new_cent_id,:]

	return Mu


# --- Make nice plots of clusters ---
# 
# 	Input:
# 		-numClasses: number of classes (the k in k-means)
# 		-X: the data (points should be rows)
# 		-labels: the labels for the rows in X
def plot_clusters(numClasses,X,labels,save=False):
	for k in xrange(numClasses):
		plt.plot(X[labels==k,0],X[labels==k,1],'o')

	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.title('K = ' + str(numClasses))
	if save:
		plt.savefig('figures/kmeans_2-3b_k' + str(numClasses) + '.png')
	# plt.show()

# --- Make nice plots of clusters for GMM ---
# 
# 	Input:
# 		-numClasses: number of classes (the k in k-means)
# 		-X: the data (points should be rows)
# 		-labels: the labels for the rows in X
# 		-Mu: centers of clusters
# 		-S: list of covariance matrices
def plot_gmm(numClasses,X,labels,Mu,S,save=False):
	# Plot the points
	for k in xrange(numClasses):
		plt.plot(X[labels==k,0],X[labels==k,1],'o')

	plt.xlabel('x1')
	plt.ylabel('x2')

	# Plot the means
	plt.plot(Mu[:,0],Mu[:,1],'kx')

	# Plot the covariance ellipses
#----------------------------------------------------------------------------------------
# 									Numerical tests
#----------------------------------------------------------------------------------------

# Read in the data
data = pd.read_csv('2DGaussianMixture.csv').as_matrix()
true_labels = data[:,0]
X = data[:,1:]

# ---------------------------
# Part (b)
# ---------------------------

# # Apply k-means
# ks = [2,3,5,10,15,20]

# for k in ks:
# 	labels, mu, err = kmeans(k,X)
# 	plot_clusters(k,X,labels,True)


# ---------------------------
# Part (c)
# ---------------------------

# residual_vec = np.empty(20)

# # Run k-means 20 times with different initializations, plotting centers
# for i in xrange(20):
# 	labels, mu, err = kmeans(3,X)
# 	plt.plot(mu[:,0],mu[:,1],'kx')
# 	residual_vec[i] = err[-1]

# # Plt data points and get residual statistics
# plt.scatter(X[:,0],X[:,1], marker='o', color='0.85')
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.title('Centers generated in 20 runs of Lloyd\'s algorithm')
# plt.savefig('figures/kmeans_2-3c_centers.png')
# # plt.show()

# print "Within-cluster sums of squares statistics:\n"
# print "Min:\t%f" %residual_vec.min()
# print "Mean:\t%f" %np.mean(residual_vec)
# print "Std:\t%f" %np.std(residual_vec)


# ---------------------------
# Part (d)
# ---------------------------

# residual_vec = np.empty(20)

# # Run k-means 20 times with different initializations, plotting centers
# for i in xrange(20):
# 	mu_init = kmeans_pp_init(3,X)
# 	labels, mu, err = kmeans(3,X,Mu=mu_init)
# 	plt.plot(mu[:,0],mu[:,1],'kx')
# 	residual_vec[i] = err[-1]

# # Plt data points and get residual statistics
# plt.scatter(X[:,0],X[:,1], marker='o', color='0.85')
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.title('Centers generated in 20 runs of K-means++')
# plt.savefig('figures/kmeans_2-3d_centers.png')
# # plt.show()

# print "Within-cluster sums of squares statistics (K-means++):\n"
# print "Min:\t%f" %residual_vec.min()
# print "Mean:\t%f" %np.mean(residual_vec)
# print "Std:\t%f" %np.std(residual_vec)


# ---------------------------
# Part (g)
# ---------------------------

labels, mu, S, log_like = em_GMM(3,X,kmeans_pp_init(3,X))

plot_clusters(3,X,labels,False)

plt.show()


plt.figure()
plt.plot(range(len(log_like)),log_like,'b-o')
plt.xlabel('Iteration')
plt.ylabel('Log likelihood')
plt.title('Log likelihood as a function of iterations of EM')
plt.show()
