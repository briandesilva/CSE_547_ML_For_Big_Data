# CSE 547 Homework 4
# Brian de Silva
# Spring 2017


# ------------------------------------------------------------------------------
# 							Thompson Sampling
# ------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
# np.random.beta(a, b)
# np.random.binomial(1, p)


class ThompsonSampling():
	"""
	A class for simulating Thompson sampling for solving the K-armed bandit problem
	Uses Beta(1,1) priors for arms and assumes arm i is Bernoulli(probs[i]) 
	"""
	def __init__(self,k,probs,T,output_times=[]):
		self.k = k 					# Number of arms
		self.probs = np.array(probs)# Probabilities associated with each arm
		self.T = T 					# Number of iterations to run
		self.output_times = np.array(output_times)
		self.t = 1
		self.successes = np.ones(self.k)
		self.failures = np.ones(self.k)
		self.reward = 0
		self.avg_regret = np.ones(self.T) * max(self.probs)

	# Function to choose an arm, pull it, and update its posterior
	def pull_arm(self):
		# Choose which arm to pull
		a = 0
		mx = 0
		for i in xrange(self.k):
			# Draw sample
			theta = np.random.beta(self.successes[i],self.failures[i])
			if theta > mx:
				mx = theta
				a = i

		# Pull arm and update posterior/reward
		r = np.random.binomial(1,self.probs[a])
		if r == 1:
			self.successes[a] += 1
			self.reward += 1
		else:
			self.failures[a] += 1

		# Update average regret
		self.avg_regret[self.t-1] -= (1. * self.reward ) / self.t

		self.t += 1

	# Get the means of the posteriors
	def get_means(self):
		return 1 / (1 + 1. * self.failures / self.successes)

	# Get the variances of the posterior distributions
	def get_var(self):
		return (self.successes * self.failures) / ((self.successes + self.failures + 1.)(self.successes + self.failures)**2)


	# Run the actual Thompson sampling scheme
	def run(self):
		meanMat = np.empty((self.k,len(self.output_times)))
		varMat = np.empty((self.k,len(self.output_times)))

		for tt in xrange(self.T):
			self.pull_arm()

			if tt in self.output_times:
				meanMat[:,tt] = self.get_means()
				varMat[:,tt] = self.get_var()


		return (meanMat, varMat)










# Main code
k = 5
probs = [1/6., 1/2., 2/3., 3/4., 5/6.]
T = 300


TS = ThompsonSampling(k,probs,T)
(meanMat,varMat) = TS.run()

plt.figure()
plt.plot(TS.avg_regret)
plt.xlabel('t')
plt.ylabel('Average regret')
plt.title('Average regret')






plt.show()
