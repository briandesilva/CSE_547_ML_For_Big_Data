import os
import subprocess
from operator import itemgetter

from Cluster import Cluster
import HDFSUtil


KMEANS_HDFS_PATH = "/user/brian/kmeans"
HADOOP_PREFIX = "/usr/local/hadoop-1.2.1"


class KmeansMRDriver(object):

    def __init__(self, numClusters):
        """ Creates a driver running kmeans with given number of clusters. """
        self.K = numClusters
        self.dictionary = []


    def run_iter(self, inputpath, outputpath, cache_path, iteration):
        """
        Run an iteration of Kmeans. Take tfidf input from inputpath, take the
        current cluster centers from cachepath, and output the new cluster
        centers and within cluster distance into outputpath.
        """
        hadoop = os.path.join(HADOOP_PREFIX, "bin", "hadoop")
        jarfile = os.path.join(HADOOP_PREFIX, "contrib", "streaming", "hadoop-streaming-1.2.1.jar")
        curdirname = os.path.dirname(os.path.realpath(__file__))
        mapper_file = "KmeansMapper.py"
        reducer_file = "KmeansReducer.py"
        other_files = ["HDFSUtil.py", "Cluster.py", "Document.py", "MathUtil.py"]
        files = []
        files.append(os.path.join(curdirname, mapper_file))
        files.append(os.path.join(curdirname, reducer_file))
        for filename in other_files:
            files.append(os.path.join(curdirname, filename))
        # Call out to hadoop streaming with a bunch of arguments
        args = [hadoop, "jar", jarfile,
                "-mapper", "KmeansMapper.py", "-reducer", "KmeansReducer.py",
                "-input", inputpath, "-output", outputpath,
                "-outputformat", "org.apache.hadoop.mapred.lib.MultipleTextOutputFormat",
                "-cmdenv", "numClusters=" + str(self.K),
                "-cmdenv", "kmeansIter="  + str(iteration),
                "-cmdenv", "kmeansHDFS=" + KMEANS_HDFS_PATH,
                "-cmdenv", "hadoopPrefix=" + HADOOP_PREFIX]
        for f in files:
            args.extend(["-file", f])
        subprocess.call(args)


    def print_distances(self, distances):
        """"
        Print out the total distance within each cluster.
        @param distances
        """
        print "=================== Squared distances within the clusters ================="
        totaldistance = 0.0
        string = ""
        for s in distances:
            splits = s.split("|")
            string += "Cluster " + splits[0] + ": " + splits[1] + "\n"
            totaldistance += float(splits[1])
        string += "Total squared distance: " + str(totaldistance)
        print string


    def print_clusters(self, clusters):
        """
        Print out the top 10 words from each cluster.
        @param clusters
        """
        print "=================== Top words in the clusters ================= "
        for c in clusters:
            string = "Cluster " + str(c.uid) + "| "
            entries = [[key, val] for key, val in c.tfidf.iteritems()]
            sorted_entries = sorted(entries, key=itemgetter(1), reverse=True)
            # print the first 10 words and values
            string += ", ".join([(":").join([self.dictionary[entry[0]], str(entry[1])]) for entry in sorted_entries[:10]])
            print string
            print "========================================================= "


    def load_dictionary(self, path):
        """
        Load the dictionary from hdfs path.
        @param path
        """
        print "Loading dictionary from: %s" % path
        self.dictionary = []
        for line in HDFSUtil.read_lines(path, hadoop_prefix=HADOOP_PREFIX):
            self.dictionary.append(line.split(" ")[0])
        print "Succeeded in loading dictionary."


    def load_cluster(self, path, initial=False):
        """
        Load the cluster centers from hdfs path.
        @param path
        """
        clusters = []
        for line in HDFSUtil.read_lines(path, hadoop_prefix=HADOOP_PREFIX):
            if not initial:
                if line.startswith("cluster"):
                    line = line.split("\t", 1)[1]
                else:
                    continue
            c = Cluster()
            c.read(line)
            clusters.append(c)
            print "Found cluster %d" %c.uid
        print "Number of clusters found: %d" %len(clusters)
        return clusters


    def load_distance(self, path, initial=False):
        """
        Load the distance file from hdfs path.
        @param path
        """
        distances = []
        for line in HDFSUtil.read_lines(path, hadoop_prefix=HADOOP_PREFIX):
            if not initial:
                if line.startswith("distance"):
                    line = line.split("\t", 1)[1]
                else:
                    continue
            line = line.rstrip()
            distances.append(line)
        return distances


if __name__ == '__main__':
    driver = KmeansMRDriver(20)

    dictionary_path = KMEANS_HDFS_PATH + "/dictionary.txt"
    inputpath = KMEANS_HDFS_PATH + "/tfidf.txt"

    driver.load_dictionary(dictionary_path)

    clusters = driver.load_cluster(KMEANS_HDFS_PATH + "/cluster0/cluster0.txt", initial=True)
    driver.print_clusters(clusters)

    numiter = 5
    for i in range(numiter):
        outputpath = KMEANS_HDFS_PATH + "/output/cluster" + str(i+1)
        if i == 0:
            cache = KMEANS_HDFS_PATH + "/cluster0"
        else:
            cache = KMEANS_HDFS_PATH + "/output/cluster" + str(i)

        driver.run_iter(inputpath, outputpath, cache, i+1)

        clusters = driver.load_cluster(outputpath + "/part-00000")
        driver.print_clusters(clusters)
        distances = driver.load_distance(outputpath + "/part-00000")
        driver.print_distances(distances)


# Values of the objective function at each iteration
# objvals = [143559.109398, 143369.591697, 143272.510368, 143185.946584, 143114.7753]

# Commands to plot objective function:
# import matplotlib.pyplot as plt
# objvals = [143559.109398, 143369.591697, 143272.510368, 143185.946584, 143114.7753]
# plt.plot(range(1,6),objvals,'b-o')
# plt.xlabel('Iterations of K-means')
# plt.ylabel('$J(Z,\mu)$')
# plt.title('Objective Function $J(Z,\mu)$')
# plt.show()


