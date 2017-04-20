import time

import data.DocumentData as DocumentData
import util.EvalUtil as EvalUtil
from methods.LocalitySensitiveHash import LocalitySensitiveHash
from methods.GaussianRandomProjection import GaussianRandomProjection
from kdtree.KDTree import KDTree
from analysis.TestResult import TestResult


import pdb


def test_LSH(train_docs, test_docs, D, m_vals, depth):
	"""
	Tests the query time and distance for a given data set and test set using Locality sensitive hasing
	@param train_docs: dict[int => dict[int => int/float]] list of training documents
	@param test_docs: dict[int => dict[int => int/float]] list of test documents
	@param D: int - the dimension of the data points
	@param m_vals: [float] - a set of m values to test (number of projections)
	@param depth: float - Maximum Hamming distance of bins that are to be checked from current bin
	@return [TestResult] array of objects of class TestResult, which has the average time and distance for a single query
	"""
	times = [None] * len(m_vals)
	n = len(test_docs)
	for k in xrange(len(m_vals)):
		print "Forming LSH table with %d projections..."%m_vals[k]
		lsh = LocalitySensitiveHash(train_docs,D,m_vals[k])
		print "Done."

		print "Computing average lookup time and distance to nearest neighbor..."
		cum_distance = 0.0
		start_time = time.clock()
		for test_doc_id, test_document in test_docs.iteritems():
			nearest = lsh.nearest_neighbor(test_document,depth)
			cum_distance += nearest.distance

		duration = time.clock() - start_time
		print "Done.\n"
		times[k] = TestResult("LSH",n,D,m_vals[k],duration / n, cum_distance / n)
		print "Average distance: %f" %(cum_distance / n)
		print "Average time: %f\n" %(duration / n)
	return times

def test_GRP(train_docs, test_docs, D, m_vals,alpha):
	"""
	Tests the query time and distance for a given data set and test set using Gaussian random projection (and a KD table)
	@param train_docs: dict[int => dict[int => int/float]] list of training documents
	@param test_docs: dict[int => dict[int => int/float]] list of test documents
	@param D: int - the dimension of the data points
	@param m_vals: [float] - a set of m values to test (number of projections)
	@param depth: float - Maximum Hamming distance of bins that are to be checked from current bin
	@return [TestResult] array of objects of class TestResult, which has the average time and distance for a single query
	"""
	times = [None] * len(m_vals)
	n = len(test_docs)
	for k in xrange(len(m_vals)):
		print "Forming GRP KD-tree with %d projections..."%m_vals[k]
		grp = GaussianRandomProjection(train_docs,D,m_vals[k])
		print "Done."

		print "Computing average lookup time and distance to nearest neighbor..."
		cum_distance = 0.0
		start_time = time.clock()
		for test_doc_id, test_document in test_docs.iteritems():
			nearest = grp.nearest_neighbor(test_document,alpha)
			cum_distance += nearest.distance

		duration = time.clock() - start_time
		print "Done.\n"
		times[k] = TestResult("GRP",n,D,m_vals[k], duration / n, cum_distance / n)
		print "Average distance: %f" %(cum_distance / n)
		print "Average time: %f\n" %(duration / n)
	return times

def test_kd_tree(train_docs, test_docs, D, alphas):
	"""
	Tests the query time and distance for the given training and testing sets
	@param D: int - the dimension of the data points
	@param alphas: [float] - a set of alphas to test
	@return [TestResult] array of objects of class TestResult, which has the average time and distance for a single query
	"""

	# Populate the tree with the training data
	print "Forming KD-tree"
	tree = KDTree(D)
	for i, document in train_docs.iteritems():
		key = [document.get(idx,0) for idx in xrange(0, D)]
		tree.insert(key, i)
	print "Done"

	times = []
	n = len(test_docs)
	for alpha in alphas:
		print "Computing average lookup time and distance to nearest neighbor for alpha = %d" %alpha
		start_time = time.clock()
		cum_dist = 0.0
		for i, test_doc in test_docs.iteritems():
			key = [test_doc.get(idx,0) for idx in xrange(0, D)]
			doc_id = tree.nearest(key, alpha)
			cum_dist += EvalUtil.distance(test_doc, train_docs[doc_id])
		duration = time.clock() - start_time
		times.append(TestResult("KDTree", n, D, alpha, duration / n, cum_dist / n))
		print "Average distance: %f" %(cum_dist / n)
		print "Average time: %f\n" %(duration / n)
	return times





if __name__ == '__main__':
    docdata = DocumentData.read_in_data("../../data/hw2/sim_docdata/sim_docdata.mtx", True)
    testdata = DocumentData.read_in_data("../../data/hw2/sim_docdata/test_docdata.mtx", True)
    print "Number of Documents: %d" % len(docdata)
    print "Number of Test Documents: %d" % len(testdata)
    D = 1000
    

    # 
    # Test the Locality Sensitive Hash Table on the Nearest neighbors problem
    # 

    # Number of projections
 #    m = [5, 10, 20]

 #    # Test LSH
 #    times_lsh = test_LSH(docdata,testdata,D,m,3)

 #    # Print results
 #    for k in xrange(len(m)):
 #    	print "LSH statistics using m = %d projections:" %m[k]
 #    	print "\tAverage query time:\t%f" %times_lsh[k].avg_time
 #    	print "\tAverage distance:\t%f" %times_lsh[k].avg_distance


 #    # Test Gaussian random projections fed into KD-tree
 #    times_grp = test_GRP(docdata,testdata,D,m,1)

 #    # Print results
	# for k in xrange(len(m)):
	# 	print "GRP statistics using m = %d projections:" %m[k]
	# 	print "\tAverage query time:\t%f" %times_grp[k].avg_time
	# 	print "\tAverage distance:\t%f" %times_grp[k].avg_distance


    # Test KD-tree
    alphas = [1, 5, 10]
    times_kdt = test_kd_tree(docdata,testdata,D,alphas)

    # Print results
    for k in xrange(len(alphas)):
    	print "KD-tree statistics using pruning parameter alpha = %d:" %alphas[k]
    	print "\tAverage query time:\t%f" %times_kdt[k].avg_time
    	print "\tAverage distance:\t%f" %times_kdt[k].avg_distance
    




