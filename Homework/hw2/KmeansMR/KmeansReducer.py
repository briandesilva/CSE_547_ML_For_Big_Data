#!/usr/bin/env python

import sys
import math
from itertools import groupby
from operator import itemgetter

from Document import Document
from Cluster import Cluster
import MathUtil


class KmeansReducer:
    """ Update the cluster center and compute the with-in class distances """

    def emit(self, key, value, separator="\t"):
        """ Emit (key, value) pair to stdout for hadoop streaming """
        print '%s%s%s' % (key, separator, value)

    def read_mapper_output(self, file, separator='\t'):
        for line in file:
            yield line.rstrip().split(separator, 1)

    def reduce(self, uid, values):
        c = Cluster()
        sqdist = 0.0
        # TODO: Put your code here!

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
