#!/usr/bin/env python

import sys
import os

from Document import Document
from Cluster import Cluster
import MathUtil
import HDFSUtil


class KmeansMapper:
    def __init__(self):
        self.K = int(os.environ.get("numClusters", 1))              # Number of clusters
        self.iteration = int(os.environ.get("kmeansIter", 1))       # Which iteration we're on
        self.kmeans_hdfs_path = os.environ.get("kmeansHDFS")
        self.hadoop_prefix = os.environ.get("hadoopPrefix")
        self.clusters = []

    def read_input(self, file):
        for line in file:
            yield line.rstrip()

    def emit(self, key, value, separator="\t"):
        """ Emit (key, value) pair to stdout for hadoop streaming """
        #print >> sys.stderr, "emitting document: %s" % key
        print '%s%s%s' % (key, separator, value)

    def main(self):
        if self.iteration == 1:
            path = self.kmeans_hdfs_path + "/cluster0/cluster0.txt"
        else:
            path = self.kmeans_hdfs_path + "/output/cluster" + str(self.iteration - 1) + "/part-00000"
        for line in HDFSUtil.read_lines(path, hadoop_prefix=self.hadoop_prefix):
            if self.iteration > 1:
                if line.startswith("cluster"):
                    line = line.split("\t", 1)[1]
                else:
                    continue
            c = Cluster()
            c.read(line)
            self.clusters.append(c)
        data = self.read_input(sys.stdin)
        for line in data:
            self.map(line)

    # Map each line (data point/value) to a key (cluster)
    def map(self, line):
        # Key is cluster - clusters stored in self.clusters
        # Value is the line
        dist = float("inf")
        key = -1
        doc = Document(line)
        for c in self.clusters:
            if MathUtil.compute_distance(doc.tfidf,c.tfidf) < dist:
                key = c.uid

        self.emit(str(key),str(doc))


if __name__ == '__main__':
    mapper = KmeansMapper()
    mapper.main()
