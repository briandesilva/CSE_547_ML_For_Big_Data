import time
import sys

from kdtree.KDTree import KDTree
import data.RandomData as RandomData
import util.EvalUtil as EvalUtil
from analysis.TestResult import TestResult


def test_kd_tree(n, D, n_test, alphas):
    """
    Tests the query time and distance for a random data set and test set
    @param n: int - the number of points of the dataset
    @param D: int - the dimension of the data points
    @param n_test: int - the number of points to test
    @param alphas: [float] - a set of alphas to test
    @return [TestResult] array of objects of class TestResult, which has the average time and distance for a single query
    """
    documents = RandomData.random_dataset(n, D)
    test_documents = RandomData.random_dataset(n_test, D)

    rand_tree = KDTree(D)
    for i, document in documents.iteritems():
        key = [document.get(idx) for idx in xrange(0, D)]
        rand_tree.insert(key, i)

    times = []
    for alpha in alphas:
        start_time = time.clock()
        cum_dist = 0.0
        for i, test_document in test_documents.iteritems():
            key = [test_document.get(idx) for idx in xrange(0, D)]
            doc_id = rand_tree.nearest(key, alpha)
            cum_dist += EvalUtil.distance(test_document, documents[doc_id])
        duration = time.clock() - start_time
        times.append(TestResult("KDTree", n, D, alpha, duration / n_test, cum_dist / n_test))
    return times


if __name__ == "__main__":
    n = 100000
    D = 1000
    n_test = 100
    alphas = [1, 1.1, 1.5, 2, 4, 10]

    times = test_kd_tree(n,D,n_test,alphas)

    test_no = 0
    for alpha in alphas:
        print "Time to find NN for %d points of %d total docs with %d features and alpha=%f: %f" %(n_test,n,D,alpha,times[test_no].avg_time)
        test_no += 1

# Takes fucking forever
