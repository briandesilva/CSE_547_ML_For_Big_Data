#!/usr/bin/env python

import sys
import math
from itertools import groupby
from operator import itemgetter

from Document import Document
from Cluster import Cluster
import MathUtil


class KmeansReducer:
    """ Update the cluster center and compute the within class distances """

    def emit(self, key, value, separator="\t"):
        """ Emit (key, value) pair to stdout for hadoop streaming """
        print '%s%s%s' % (key, separator, value)

    def read_mapper_output(self, file, separator='\t'):
        for line in file:
            yield line.rstrip().split(separator, 1)

    def reduce(self, uid, values):
        # values is a list of dictionaries?
        c = Cluster()
        c.uid = uid
        sqdist = 0.0
        average = {}

        # Compute new center
        count = 0
        for value in values:
            count +=1
            doc = Document(value)
            for key, v in doc.tfidf.items():
                average[key] = average.get(key,0.0) + v

        for key in average:
            average[key] /= count


        # Get within cluster distance
        for value in values:
            doc = Document(value)
            sqdist += MathUtil.compute_distance(average,doc.tfidf)

        # Update the cluster center
        c.tfidf = average
        # Output the cluster center into file: clusteri
        self.emit("cluster" + str(c.uid), str(c))
        # Output the within distance into file: distancei
        self.emit("distance" + str(c.uid), str(c.uid) + "|" + str(sqdist))


    def main(self):
        data = self.read_mapper_output(sys.stdin)
        for uid, values in groupby(data, itemgetter(0)):
            vals = [val[1] for val in values]
            self.reduce(uid, vals)


if __name__ == '__main__':
    reducer = KmeansReducer()
    reducer.main()
