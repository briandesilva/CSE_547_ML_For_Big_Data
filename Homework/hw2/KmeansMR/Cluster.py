class Cluster(object):
    """
    Class representing a cluster center with its mean and id.
    """

    def __init__(self):
        """
        Creates an empty cluster with uid=-1
        """
        self.uid = -1
        self.tfidf = {}


    def read(self, line):
        """
        Parse the string and fill in the cluster.
        @param line: string representation of cluster
        """
        splits = line.split("|")
        self.uid = int(splits[0])
        self.tfidf = {}
        for token in splits[1].split(","):
            pairs = token.split(":")
            self.tfidf[int(pairs[0])] = float(pairs[1])


    def __str__(self):
        """
        Stringify cluster
        """
        string = str(self.uid) + "|"
        addcomma = False
        for wordid in self.tfidf:
            if addcomma:
                string += ","
            else:
                addcomma = True
            string += str(wordid) + ":" + str(self.tfidf[wordid])
        return string
