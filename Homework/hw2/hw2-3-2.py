# CSE 547 Homework 2
# Problem 3.2 - K-means and EM on BBC data
# Brian de Silva


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.io import mmread
from scipy.stats import mode

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
# 		-true_labels is the set of actual labels
# 		-Mu is the set of initial centers
# 		-MAX_ITS is the maximum number of iterations taken before k-means terminates
# 	Output:
# 		-Y is the set of labels for points in X (labels take values 0,1,...,numClasses-1)
# 		-Mu is the set of means of the classes (centers[i] corresponds to class i)
# 		-z1_loss is the 0/1 loss at each iteration
def kmeans(numClasses,X,true_labels,Mu=None,MAX_ITS=300):
	N = X.shape[0]						# Number of data points
	Y = np.zeros(N).astype(int)			# Labels
	Yprev = np.zeros(N).astype(int)		# Labels from previous iteration
	z1_loss = np.zeros(MAX_ITS)			# 0/1 loss

	# Initialize centers as numClasses random points in data
	if Mu is None:
		Mu = X[np.random.choice(np.arange(N),numClasses,replace=False),:]

	# Store square 2-norms of data points to speed up computation
	X2 = np.sum(X**2,1)

	# Iteratively update centers until convergence
	for it in xrange(MAX_ITS):

		# Compute quantities we can reuse
		Mu2 = np.sum(Mu**2,1)

		# Assign labels and (starts at iteration 0)
		for k in xrange(N):
			dists = Mu2 + X2[k] - 2. * Mu.dot(X[k,:])
			Y[k] = np.argmin(dists)

		# 
		# Get 0/1 loss
		# 

		# Determine true labels for each mean
		meanLabels = np.empty(numClasses)
		for k in xrange(numClasses):
			meanLabels[k] = mode(true_labels[Y==k])[0][0]

		# Map the k-means labels for each point to corresponding true labels
		Y_mapped = meanLabels[Y]

		# Get 0/1 loss
		z1_loss[it] = (1.0 * np.count_nonzero(Y_mapped - true_labels)) / N

		# Compute new centers
		for k in xrange(numClasses):
			Mu[k,:] = np.mean(X[Y==k,:],0)

		# Check if we should exit
		if np.array_equal(Y,Yprev):
			break
		else:
			Yprev = np.copy(Y)

	print "%d iterations of k-means to achieve convergence." %(it+1)

	return (Y,Mu,z1_loss[:(it+1)])


# --- EM for GMM ---
# Expectation maximization Gaussian Mixture Model implementation
# 	Input:
# 		-numClasses is the number of clusters
# 		-X is the data matrix (rows are data points)
# 		-true_labels is the set of actual labels
# 		-Mu is the set of initial centers
# 		-TOL is the tolerance for how small the log likelihood can change between iterations
# 		-MAX_ITS is the maximum number of iterations taken before the algorithm terminates
# 	Output:
# 		-Y is the set of labels for points in X (labels take values 0,1,...,numClasses-1)
# 		-Mu is the set of means of the classes (centers[i] corresponds to class i)
# 		-S is a list of the covariance matrices
# 		-log_like is the log-likelihood at each iteration
# 		-z1_loss is the 0/1 loss at each iteration
def em_GMM(numClasses,X,true_labels,Mu=None,TOL=1.e-6,MAX_ITS=300):
	N = X.shape[0]						# Number of data points
	Y = np.zeros(N).astype(int)			# Labels
	log_like = np.zeros(MAX_ITS)		# Square reconstruction error
	lambduh = 0.2						# Shrinkage parameter for covariance matrices

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
		for n in xrange(N):
			for k in xrange(numClasses):
				likelihood_matrix[n,k] = pi[k] * multivariate_normal.pdf(X[n,:],mean=Mu[k,:],cov=S[k])
		
		# Store and check log-likelihood
		log_like[it] = np.sum(np.log(np.sum(likelihood_matrix,1)))
		if (it>0) and (np.abs(log_like[it] - log_like[it-1])) < TOL:
			break


		# E step:

		# Precompute denominators
		denom = np.sum(likelihood_matrix,1)

		# Compute responsibilities
		R = likelihood_matrix / denom[:,None]

		# 
		# Get 0/1 loss
		# 

		# Determine true labels for each mean
		Y = np.argmax(R,axis=1)
		meanLabels = np.empty(numClasses)
		for k in xrange(numClasses):
			meanLabels[k] = mode(true_labels[Y==k])[0][0]

		# Map the k-means labels for each point to corresponding true labels
		Y_mapped = meanLabels[Y]

		# Get 0/1 loss
		z1_loss[it] = (1.0 * np.count_nonzero(Y_mapped - true_labels)) / N

		# M step:

		# Update means
		Nk = np.sum(R,0)		# Weighted number of points assigned to each class
		Mu = R.T.dot(X) / Nk[:,None]

		# Update covariance matrices
		for k in xrange(numClasses):
			XMM = X - Mu[k,:]
			S[k] = np.zeros((X.shape[1],X.shape[1]))
			for n in xrange(N):
				S[k] += R[n,k] * np.outer(XMM[n,:],XMM[n,:])
			S[k] /= Nk[k]
			S[k] = (1.-lambduh) * S[k] + lambduh * np.identity(X.shape[1])

		# Update responsibiliites
		pi = Nk / N

	print "EM for GMM converged after %d iterations." %it
	# Set some output quantities
	Y = np.argmax(R,axis=1)
	return log_like[:(it+1)], z1_loss[:(it+1)]


#----------------------------------------------------------------------------------------
# 									Numerical tests
#----------------------------------------------------------------------------------------



# -----------------------------------------------
# 				Part 2 - BBC data
# -----------------------------------------------

# Read in the data
f_mat = mmread('bbc_data/bbc.mtx').tocsr()			# term-document frequency matrix
classes = pd.read_csv('bbc_data/bbc.classes',sep=' ',header=None).as_matrix()		# The document classes
file = open('bbc_data/bbc.terms','r')
terms = file.read().splitlines()					# The terms for each row in f_mat
file.close()

# Get the centers with which to initialize k-means and EM
centers = np.loadtxt('bbc_data/bbc.centers',delimiter=" ")

# ---------------------------
# Part (a)
# ---------------------------

# Compute tfidf
max_f = f_mat.max(axis=0).toarray()								# Max frequency of any word for each doc
tfidf = np.array(f_mat / max_f)									# Term frequency matrix
D = f_mat.shape[1]												# Number of documents in corpus
idf = np.array(np.log(D / np.sum(f_mat != 0.0, axis=1)))		# Inverse document frequency
tfidf = tfidf * idf										# tf-idf

# Compute average tfidf across the classes
numClasses = np.max(classes[:,1]) + 1
avg_tfidf = np.empty((tfidf.shape[0],numClasses))
for ci in xrange(numClasses):
	class_inds = (classes[:,1] == ci)
	avg_tfidf[:,ci] = np.sum(tfidf[:,class_inds],axis=1) / np.sum(class_inds)

	# Report 5 terms with highest average tf-idf for each class
	temp = np.argsort(avg_tfidf[:,ci])
	print "For class %d the five terms with highest average tf-idf are:" %ci
	print ""
	for k in xrange(1,6):
		print "\t%s\t&\t%f" %(terms[temp[-k]], avg_tfidf[temp[-k],ci])


# ---------------------------
# Part (b)
# ---------------------------

# Apply K-means
labels, mu, z1_loss = kmeans(numClasses,tfidf.T,classes[:,1],centers)

# Plot 0/1 loss against number of iterations
plt.plot(range(len(z1_loss)),z1_loss,'b-o')
plt.xlabel('Iterations')
plt.ylabel('0/1 Loss')
plt.title('0/1 loss for K-means as a function of iterations')
plt.savefig('figures/kmeans_2-3-2b')
plt.show()


# ---------------------------
# Part (c)
# ---------------------------

# Apply EM for GMM
log_like, z1_loss = em_GMM(numClasses,tfidf.T,classes[:,1],Mu=centers,MAX_ITS=5)


# Plot 0/1 loss against number of iterations
plt.plot(range(len(z1_loss)),z1_loss,'b-o')
plt.xlabel('Iterations')
plt.ylabel('0/1 Loss')
plt.title('0/1 loss for EM for GMM as a function of iterations')
plt.savefig('figures/EM_z1loss_2-3-2c')
plt.show()

# Plot log-likelihood as a function of iterations
plt.figure()
plt.plot(range(len(log_like)),log_like,'b-o')
plt.xlabel('Iteration')
plt.ylabel('EM Log-likelihood')
plt.title('Log-likelihood as a function of iterations of EM')
plt.savefig('figures/EM_likelihood_2-3-2c')
plt.show()
