class Document(object):
    """
    Class representing a document with with its tfidf and id.
    """

    def __init__(self, line):
        """
        Document constructor
        @param line: string representation of document
        """
        self.uid = -1
        self.tfidf = {}
        if not line:
            return
        splits = line.split("|")
        self.uid = splits[0]
        tokens = splits[1].split(",")
        for token in tokens:
            pairs = token.split(":")
            self.tfidf[int(pairs[0])] = float(pairs[1])


    def __str__(self):
        """
        Stringify document
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
