import methods.Helper as Helper
from kdtree.KDTree import KDTree
from methods.NeighborDistance import NeighborDistance
import util.EvalUtil as EvalUtil


class GaussianRandomProjection(object):
    """
    @ivar documents: dict[int => dict[int => int/float]] list of documents
    @ivar D: int - dimension of vectors
    @ivar m: int - number of random projections
    @ivar projection_vectors: [[float]] - the projection vectors
    @ivar kdt: methods.KDTree - a KD-tree instance
    """

    def __init__(self, documents, D, m):
        """
        Creates a GaussianRandomProjection with the specified dimension and
        number of random projections
        @param documents: dict[int => dict[int => int/float]] - the documents
        @param D: int - dimension
        @param m: int - number of random projections
        """
        self.documents = documents
        self.D = D
        self.m = m
        self.projection_vectors = Helper.create_projection_vectors(D, m)
        self.kdt = KDTree(m)
        for doc_id, doc in documents.iteritems():
            self.kdt.insert(self.hash_document(doc), doc_id)


    def nearest_neighbor(self, document, alpha):
        """
        Finds the approximate nearest neighbor for given document.
        @param document: dict[int => int/float] - document represented as dictionary of word ids => counts
        @param alpha: float - alpha for approximate k-nn
        """
        hashed_document = self.hash_document(document)
        nearest_id = self.kdt.nearest(hashed_document, alpha)
        distance = EvalUtil.distance(document, self.documents[nearest_id])
        return NeighborDistance(nearest_id, distance)


    def hash_document(self, document):
        """
        Hashes a document using the random projections
        @param document: dict[int => int/float] - document represented as dictionary of word ids => counts
        """
        # hashed_document = [0.0 for _ in self.m]
        # # TODO: hash/project the document onto the "m" gaussian random projections
        # raise Exception("Please implement the GaussianRandomProjection.hash_document method")
        # return hashed_document

        return [self.project_document(document,self.projection_vectors[i]) for i in xrange(self.m)]



    def project_document(self, document, vector):
        """
        Projects a document onto a vector.
        @param document: dict[int => int/float] - document represented as dictionary of word ids => counts
        @param vector: [float] - a vector on which to project the document
        """
        dotprod = 0.0
        for word in document:
            dotprod += document[word] * vector[word-1]
        return dotprod
