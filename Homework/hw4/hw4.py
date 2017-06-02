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
		self.k            = k 					# Number of arms
		self.probs        = np.array(probs)     # Probabilities associated with each arm
		self.T            = T 					# Number of iterations to run
		self.output_times = np.array(output_times)
		self.t            = 1
		self.successes    = np.ones(self.k)
		self.failures     = np.ones(self.k)
		self.reward       = 0
		self.avg_regret   = np.ones(self.T) * max(self.probs)
		self.N 			  = np.zeros((k,self.T))

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
				a  = i

		# Pull arm and update posterior/reward
		if self.t > 1:
			self.N[:,self.t-1] = self.N[:,self.t-2]
		self.N[a,self.t-1] += 1
		r = np.random.binomial(1,self.probs[a])
		if r == 1:
			self.successes[a] += 1
			self.reward       += 1
		else:
			self.failures[a]  += 1

		# Update average regret
		self.avg_regret[self.t-1] -= (1. * self.reward ) / self.t

		self.t += 1

	# Get the means of the posteriors
	def get_means(self):
		return 1 / (1 + 1. * self.failures / self.successes)

	# Get the variances of the posterior distributions
	def get_var(self):
		return (self.successes * self.failures) / ((self.successes + self.failures + 1.) * (self.successes + self.failures)**2)

	# Plot the fraction of the time spent on each arm
	def plot_N(self):
		plt.figure()
		t = np.arange(1,self.T+1)
		for a in xrange(self.k):
			plt.plot(t,self.N[a,:] / t,label="Arm " + str(a+1))
		plt.xlabel('t')
		plt.ylabel('$N_{a,t}/t$')
		plt.title('Fraction of time spent on each arm')
		plt.legend()
		plt.savefig("figures/N_at.png")

	# Run the actual Thompson sampling scheme
	def run(self):

		for tt in xrange(self.T):
			self.pull_arm()

			if tt+1 in self.output_times:
				self.plot_ci()

		self.plot_N()



	# Plot "confidence intervals" for the arms
	def plot_ci(self):
		means     = self.get_means()
		variances = self.get_var()

		plt.figure()
		a = np.arange(1,self.k+1)
		plt.plot(a,self.probs,'*',label="$\mu_a$")
		plt.errorbar(a,means,fmt='o',yerr=variances*5,label="$\hat\mu_{a,t}$")
		plt.xlabel('Arm')
		plt.ylabel('$\mu$')
		plt.legend()
		plt.title("Confidence intervals at t = " + str(self.t-1))
		plt.savefig("figures/CI_t" + str(self.t-1) + ".png")







# ------------------------------------------------------------------------------
# 								Main Code
# ------------------------------------------------------------------------------




# Parameters
k            = 5	                         	# Number of arms (each with Bernoulli(p) rewards)
probs        = [1/6., 1/2., 2/3., 3/4., 5/6.]	# Probabilities for each arm
T            = 10000                            	# Number of iterations
output_times = [5, 25, 500, 1000]


# Perform the Sampling
TS = ThompsonSampling(k,probs,T,output_times)
TS.run()

# Plot the average regret
plt.figure()
plt.plot(TS.avg_regret)
plt.xlabel('t')
plt.ylabel('Average regret')
plt.title('Average regret')
plt.savefig('figures/avg_regret.png')
# plt.show()


# Determine the first time t where N_{5,t}/t is above 0.95 and it
# stays above it for at least 10 steps in a row
inds = np.nonzero(TS.N[4,:] / np.arange(1,T+1) > 0.95)[0]

count = 0
start = 0
doneFlag = False
for i in xrange(len(inds)-1):
	start = inds[i]
	count = 0
	while inds[i] == inds[i+1] - 1:
		count += 1
		i += 1
		if count == 10:
			doneFlag = True
			break
	if doneFlag:
		break
if doneFlag:
	print "First time: t = %d" %(start + 1)
else:
	print "No first time."