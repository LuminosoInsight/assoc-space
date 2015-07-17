import numpy as np

class RBVectorSpace:
    """
    RBVectorSpace implements the redundant bit vector data structure. This data
    structure allows fast approximate locality based searching of high
    dimensional vector spaces.
    """
    # All the vectors in the current use case of RBVectorSpace are normalized
    # can we use that to our advantage?

    def __init__(self, vectors, radius=0.3, buckets=5, dimensions=None,
                    verbose=True):
        """
        Creates a RBVectorSpace.

        `vectors` is the list of n-dimensional vectors to be indexed.

        `radius` is the side length of the indexing hypercubes.

        `buckets` is the number of buckets per dimension

        Note: maximizing cosine similarity coincides with minimizing the
        euclidean distance if all the vectors are l2 normalized.
        """
        self.vectors = vectors

        if dimensions is None:
            dimensions = vectors.shape[1]
        self.dimensions = dimensions

        vectors = vectors[:,:dimensions]

        if verbose:
            print("Generating bucket edges")

        #generate edges of buckets
        edges = np.concatenate((vectors+radius, vectors-radius), axis=0)

        if verbose:
            print("Sorting edges")

        edges.sort(axis=0)

        edges = edges[
            np.linspace(0, edges.shape[0]-1, num=buckets+1, dtype=int)
        ]

        self.edges = edges

        if verbose:
            print("Computing bit vectors")

        bit_buckets = np.logical_and(
            edges.T[:,:-1,np.newaxis] <= vectors.T[:,np.newaxis,:],
            vectors.T[:,np.newaxis,:] <= edges.T[:,1:,np.newaxis]
        )

        self.buckets = bit_buckets

    def find_nearest(self, vector, verbose=True):
        """
        Find the element of the vector space that is closest to `vector`. This
        algorithm is approximate and may sometimes return an incorrect result.
        """
        #TODO
        vector = vector[:self.dimensions]

        if verbose:
            print("Computing indices")

        indices = np.nonzero(
            np.logical_and(
                self.edges[:-1] <= vector,
                vector <= self.edges[1:]
            ).T
        )

        buckets = self.buckets[indices]

        if verbose:
            print("Intersecting buckets")

        nearby = np.ones(buckets.shape[1], dtype=bool)
        for bucket in buckets:
            nearby &= bucket

        if verbose:
            print("Performing reduced linear search")

        return self._linear_search(vector, self.vectors[nearby])


    def _linear_search(self, vector, vectors):
        """
        Finds the element of `vectors` that is closest to `vector`. This
        computes the distance between each element of `vectors` and `vector`
        and returns the element with the smallest distance.
        """
        return vectors[np.argmin(np.sum((vectors-vector)**2, axis=1))]
