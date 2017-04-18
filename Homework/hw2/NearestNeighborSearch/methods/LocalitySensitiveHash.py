import math

import methods.Helper as Helper
from methods.NeighborDistance import NeighborDistance
import util.EvalUtil as EvalUtil

# Debugger
import pdb


class LocalitySensitiveHash(object):
    """
    @ivar documents: dict[int => dict[int => int/float]] list of documents
    @ivar D: int - dimension of documents / vectors (number of unique words)
    @ivar m: int - number of random projections
    @ivar projection_vectors: [[float]] - the projection vectors
    @ivar hashed_documents: dict[int => set(int)] - hash data structure for documents
    """

    def __init__(self, documents, D, m):
        """
        Creates a LocalitySensitiveHash with the specified dimension and number
        of random projections
        @param documents: dict[int => dict[int => int/float]] - the documents
        @param D: int - dimension
        @param m: int - number of projections / hashes
        """
        self.documents = documents
        self.D = D
        self.m = m
        self.projection_vectors = Helper.create_projection_vectors(D, m)
        self.build_hashed_documents()


    def build_hashed_documents(self):
        """
        Builds the hash table of documents.
        """
        self.hashed_documents = dict()
        for doc_id, document in self.documents.iteritems():
            lsh_bin = self.get_bin(document)
            if not self.hashed_documents.has_key(lsh_bin):
                self.hashed_documents[lsh_bin] = set()
            self.hashed_documents[lsh_bin].add(doc_id)


    def nearest_neighbor(self, document, depth):
        """
        Gets the (approximate) nearest neighbor to the given document
        @param document: dict[int => int/float] - a document
        @param depth: int - the maximum number of bits to change concurrently
        """
        hashed_document = self.hash_document(document)
        nearest = self._nearest_neighbor(document, hashed_document, None, depth, 0)
        return nearest


    def _nearest_neighbor(self, document, hashed_document, cur_nearest, depth, next_index):
        """
        Helper function to get the (approximate) nearest neighbor to the given document
        @param document: dict[int => int/float] - a document
        @param hashed_document: [bool] - hashed  document
        @param cur_nearest: NeighborDistance - the currently (approximately) closest neighbor
        @param depth: int - the maximum number of bits to change concurrently
        @param next_index: int - the next bin on which to potentially flip a bit
        """
        if depth < 0:
            return cur_nearest
        if cur_nearest is None:
            cur_nearest = NeighborDistance(0, float("inf"))
        self.check_bin(document, hashed_document, cur_nearest)
        if depth > 0:
            # check the bins one away from the current bin
            # if we still have more depth to go
            for j in xrange(next_index, self.m):
                hashed_document[j] = not hashed_document[j]
                self._nearest_neighbor(document, hashed_document, cur_nearest, depth - 1, j + 1)
                hashed_document[j] = not hashed_document[j]
        return cur_nearest


    def check_bin(self, document, hashed_document, cur_nearest):
        """
        Checks the documents that are hashed to the given bin and updates with
        nearest neighbor found.
        @param document: dict[int => int/float] - list of documents
        @param hashed_document: [bool] - hashed document
        @param cur_nearest: NeighborDistance - the currently (approximately) nearest neighbor
        """
        # pdb.set_trace()
        for hashed_doc_id in self.hashed_documents.get(self.get_bin(document),[]):
        #[self.get_bin(document)]:

            # Compute distance between document and each doc in the bin
            # Note: if the document is in the dataset it will be its own nearest neighbor
            dist = EvalUtil.distance(document,self.documents[hashed_doc_id])
            # dist = 0.0
            # cur_doc = self.documents[hashed_doc_id]
            # for key in set(document.keys()).union(set(cur_doc.keys())):
            #     dist += math.pow(document.get(key,0.0) - cur_doc.get(key,0.0),2)
            # dist = math.sqrt(dist);

            # Check if this hashed_doc_id is the nearest neighbor
            if dist < cur_nearest.distance:
                cur_nearest.doc_id = hashed_doc_id
                cur_nearest.distance = dist


    def get_bin(self, document):
        """
        Gets the bin where a document should be stored.
        @param document: dict[int => int/float] - a document
        """
        return self.convert_boolean_array_to_integer(self.hash_document(document))


    def hash_document(self, document):
        """
        Hashes a document to a boolean array using the set of projection vectors
        @param document: dict[int => int/float] - a document
        """
        return [self.project_document(document,self.projection_vectors[i]) for i in xrange(self.m)]


    def project_document(self, document, vector):
        """
        Projects a document onto a projection vector for a boolean result.
        @param document: dict[int => int/float] - a document
        @param vector: [float] - a projection vector
        """

        # Note: word ids are between 1 and D (inclusive)
        dotprod = 0.0
        for word in document:
            dotprod += document[word] * vector[word-1]

        if dotprod < 0:
            return False
        return True

    def convert_boolean_array_to_integer(self, bool_array):
        """
        Converts a boolean array into the corresponding integer value.
        @param bool_array: [bool] - array of boolean values
        """
        value = 0
        for i, val in enumerate(bool_array):
            if val:
                value += math.pow(2, i)
        return int(value)
