from __future__ import print_function
import numpy as np
from scipy.sparse import coo_matrix, spdiags
from collections import defaultdict
import os

import logging
logger = logging.getLogger(__name__)

from . import eigenmath
from .compat import values, izip, zip, FileNotFoundError
from .util import lazy_property


# A version of ordered_set.OrderedSet specialized to our purposes.  It eschews
# some of the type checking that we don't use, and includes code which defers
# creating the labels-to-indices dict until required.
class LabelSet(object):
    """
    Set of (unique) labels for an external sequence (such as a matrix).

    The interface is a blend of the list interface and the set interface; for
    example, index() returns the index of an item, but the method to add a new
    item is add(), to emphasize that the new item may already be in the set.
    """

    def __init__(self, items=None):
        """
        Constructor.

        Optionally pass a list of unique items; it will be stored by reference.
        """
        if items is not None:
            self.items = items
        else:
            self.items = []

    def __contains__(self, x):
        return x in self.indices

    def __eq__(self, other):
        return self.items == other.items

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def __repr__(self):
        if len(self) < 10:
            return 'LabelSet(%r)' % self.items
        else:
            return '<LabelSet of %d items like %r>' % (len(self), self[0])

    def add(self, x):
        """
        Add the given item, returning its index.
        """
        if x not in self.indices:
            self.indices[x] = len(self.indices)
            self.items.append(x)
        return self.indices[x]

    def copy(self):
        """
        Make a copy.
        """
        result = self.__class__(self.items[:])
        if 'indices' in self.__dict__:
            result.indices = self.indices.copy()
        return result

    def index(self, x):
        """
        Return the index of the given item (possibly raising KeyError).
        """
        return self.indices[x]

    @lazy_property
    def indices(self):
        """
        A dictionary from labels to indices.
        """
        return {x: i for i, x in enumerate(self.items)}

    def merge(self, other):
        """
        Copy this LabelSet, then merge another LabelSet into the copy.

        Returns the copy and the indices for the labels that were merged in.
        """
        merged = self.copy()
        return merged, [merged.add(x) for x in other.items]

    def merge_many(self, others):
        """
        Copy this LabelSet, then merge an arbitrary number of LabelSets into
        the copy. Return the merged labels, and a list of indices for every
        merged LabelSet, indicating where its items are in the merged result.
        """
        merged = self.copy()
        index_lists = []
        for other_labels in others:
            index_lists.append([merged.add(x) for x in other_labels.items])
        return merged, index_lists


class SparseEntryStorage(object):
    """
    A helper class that stores entries of a labeled sparse matrix in
    an efficient format.
    """

    def __init__(self):
        self.labels = LabelSet()
        self.entries = defaultdict(float)
        self.total_weight = 0.

    def add_entries(self, entries):
        """
        Add triples of the form (value, row_label, col_label).
        """
        for value, row_label, col_label in entries:
            row, col = (self.labels.add(row_label), self.labels.add(col_label))
            self.entries[(row, col)] += value
            self.total_weight += abs(value)

    def get_matrix_and_metadata(self):
        """
        Create a symmetric SciPy matrix from the entries that have been stored
        here.

        Returns a triple of (matrix, labels, weight), representing the
        following:

        - matrix, the sparse matrix in SciPy coo_matrix format
        - labels, the LabelSet indicating what the rows and columns mean
        - total_weight, the sum of the absolute values of entries in the matrix
        """
        # The indices array ends up with the wrong shape if there are no
        # entries at all
        if not self.labels:
            matrix = coo_matrix((0, 0), dtype='d')
        else:
            # Borrowed from scipy.sparse.dok_matrix.tocoo()
            data = np.asarray(values(self.entries), dtype='d')
            indices = np.asarray(list(self.entries), dtype=np.intc).T
            size = len(self.labels)
            matrix = coo_matrix((data, indices), shape=(size, size))

        return matrix + matrix.T, self.labels, self.total_weight


class AssocSpace(object):
    '''
    A reduced-dimensionality representation of a set of associations.

    Operationally, an AssocSpace takes a term or weighted set of terms and
    returns a low (e.g. 150) dimensional vector such that dot products between
    different vectors provide a meaningful measure of the association between
    the terms.  The AssocSpace class provides methods to compute these vectors
    and to create AssocSpaces from provided associations.

    AssocSpaces are immutable; operations that might mutate the AssocSpace
    return a copy instead.  However, the copies may share data, either as the
    same objects or as NumPy views, with the original AssocSpace.  Likewise,
    data provided to the AssocSpace constructor is stored by reference.  Thus,
    be careful when modifying objects that may be part of an AssocSpace.
    '''

    # ===============================
    # Constructor and factory methods
    # ===============================
    def __init__(self, u, sigma, labels, assoc=None):
        '''
        Creates an AssocSpace from the matrix of eigenvectors U, the vector
        of eigenvalues sigma, and a LabelSet of labels.

        If the spectrally associated matrix was previously saved, it can be
        provided as an optional argument to avoid future computation.
        '''
        # Perform some sanity checks
        if sigma[0] == 0:
            raise ValueError('Cannot create AssocSpace with all-zero sigma.')
        if u.shape != (len(labels), len(sigma)):
            raise ValueError('Shape %s does not match expected %s' %
                             (u.shape, (len(labels), len(sigma))))
        if not (sigma[:-1] >= sigma[1:]).all():
            raise ValueError('Eigenvalues were not properly sorted.')

        if assoc is not None:
            self.assoc = assoc
        self.u = u
        self.sigma = sigma / sigma[0]
        self.k = len(sigma)
        self.labels = labels

        # This is a cache of looked-up rows, for speed
        self._row_cache = {}

    @classmethod
    def from_matrix(cls, matrix, labels, k, offset_weight=8e-6,
                    matrix_sum=None, strip_a0=True, normalize_gm=True):
        '''
        Build an AssocSpace from a SciPy sparse matrix and a LabelSet.

        Pass k to specify the number of dimensions; otherwise a value will be
        chosen for you based on the size of the matrix.

        strip_a0=True (on by default) removes the first eigenvector, which is
        often uninformative.

        normalize_gm=True (on by default) divides each entry by the geometric
        mean of the sum of the column and the sum of the row.  This is one
        iteration of a process that might eventually yield a Markov matrix.
        However, in order to suppress sufficiently rare terms, we add an offset
        to the row and column sums computed from the number of dimensions and
        the overall sum of the matrix.
        '''
        # Immediately reject empty inputs
        if not labels:
            return None

        if matrix_sum is None:
            sums = matrix.sum(0)
            matrix_sum = np.sum(sums)

        logger.info('Building space with k=%d (sum=%.6f).' % (k, matrix_sum))

        if normalize_gm:
            offset = matrix_sum * offset_weight
            normalizer = spdiags(1.0 / np.sqrt(sums + offset), 0,
                                 matrix.shape[0], matrix.shape[0])
            matrix = normalizer * matrix * normalizer

        u, s = eigenmath.eigensystem(matrix, k=k, strip_a0=strip_a0)

        # This ensures that the normalization step is sane
        if s.shape[0] == 0 or s[0] <= 0:
            return None

        return cls(np.asarray(u, '>f4'), np.asarray(s, '>f4'), labels)

    @classmethod
    def from_entries(cls, entries, **kwargs):
        """
        Build an AssocSpace out of its sparse labeled entries, as triples
        in (value, row_label, col_label) form.

        Returns None if there are no entries.

        See from_matrix() for optional arguments.
        """
        bucket = SparseEntryStorage()
        bucket.add_entries(entries)
        matrix, labels, weight = bucket.get_matrix_and_metadata()
        return cls.from_matrix(matrix, labels, **kwargs)

    @lazy_property
    def assoc(self):
        """
        The spectrally associated matrix, i.e. U e^(S/2) row-normalized.
        """
        # S/2, you say?  Why?
        # For a full explanation, see the math documentation on braindump.
        # The short version: when we generate vectors for two things and dot
        # them together, both of them will have an e^(S/2) in them, meaning
        # that between them they have the full e^S.
        unnormalized = np.multiply(self.u, np.exp(self.sigma / 2))
        return eigenmath.normalize_rows(unnormalized, offset=1e-4)

    # ==========================
    # Loading and saving on disk
    # ==========================
    def save_dir(self, directory, save_assoc=True):
        """
        Given an assoc_space object, save its parts to a directory.

        The files created are:
          * u.npy
          * sigma.npy
          * labels.txt
          * assoc.npy (redundant but saved for speed; only if save_assoc is True)
        """
        np.save(os.path.join(directory, 'u.npy'), self.u)
        np.save(os.path.join(directory, 'sigma.npy'), self.sigma)
        if save_assoc:
            np.save(os.path.join(directory, 'assoc.npy'), self.assoc)
        with open(os.path.join(directory, 'labels.txt'), 'wb') as fl:
            fl.write('\n'.join(self.labels.items).encode('utf-8') + b'\n')

    @classmethod
    def load_dir(cls, directory):
        """
        Load an AssocSpace from a directory on disk.
        """
        u = np.load(os.path.join(directory, 'u.npy'), mmap_mode='r')
        sigma = np.load(os.path.join(directory, 'sigma.npy'))
        with open(os.path.join(directory, 'labels.txt'), 'rb') as fl:
            labels = LabelSet(fl.read().decode('utf-8').splitlines())

        # Load the spectrally-associated matrix if available
        try:
            assoc = np.load(os.path.join(directory, 'assoc.npy'), mmap_mode='r')
        except FileNotFoundError:
            assoc = None

        return cls(u, sigma, labels, assoc=assoc)

    # ======================
    # Querying an AssocSpace
    # ======================
    def row_named(self, term):
        """
        Get the vector for a particular term, raising KeyError if not found.
        """
        return self.assoc[self.labels.index(term)]

    def vector_from_terms(self, terms):
        """
        Get a vector representing a weighted set of terms, provided as a
        collection of (term, weight) tuples.  Note that this does not normalize
        the rows of U e^(S/2) before taking their weighted sum; this applies a
        natural penalty to low-quality terms.
        """
        result = np.zeros((self.k,))
        for term, weight in terms:
            if term in self.labels:
                if term not in self._row_cache:
                    # Prevent the cache from growing too large
                    if len(self._row_cache) > 15000:
                        self._row_cache = {}
                    # Avoid keeping a slice of a memmap object; Numpy handles
                    # these inefficiently if you have a lot of them (especially
                    # in 1.7, but even in 1.6 or 1.8)
                    row = np.copy(self.u[self.labels.index(term)])
                    self._row_cache[term] = row
                result += self._row_cache[term] * weight
        return eigenmath.normalize(result * np.exp(self.sigma / 2))

    def terms_similar_to_vector(self, vec, sorted=True):
        """
        Find the terms most similar to the given vector, returning a list of
        (term, similarity) tuples in descending order of similarity.  (Set
        `sorted` to False to skip sorting.)
        """
        similarity = zip(self.labels, np.dot(self.assoc, vec))
        if sorted:
            similarity.sort(key=lambda x: x[1], reverse=True)
        return similarity

    def assoc_between_two_terms(self, term1, term2):
        """
        Convenience method: the dot product between the rows for two labels.
        """
        return np.dot(self.row_named(term1), self.row_named(term2))

    def all_pairs_similarity(self):
        """
        Get a NumPy matrix of the similarity measure for all pairs of labels.
        Useful for testing and debugging.

        This requires n^2 space when there are n labels, so make sure to only
        do this on small matrices.
        """
        u_sigma = np.multiply(self.u, self.sigma / np.max(self.sigma))
        return u_sigma.dot(u_sigma.T)

    def all_pairs_association(self):
        """
        Get a NumPy matrix of the association measure for all pairs of labels.
        Useful for testing and debugging.

        This requires n^2 space when there are n labels, so make sure to only
        do this on small matrices.
        """
        return self.assoc.dot(self.assoc.T)

    # ==========================
    # Transforming an AssocSpace
    # ==========================
    def filter(self, transformation):
        """
        Filter out some rows of the space, e.g. for language selection.

        Applies the following steps:
          * Apply the given transformation function, discarding labels for
            which it returns None
          * Add together rows which end up with identical labels
          * Redecompose the filtered rows to restore an orthonormal U matrix
        """
        labels = LabelSet()
        rows = []
        for label, row in izip(self.labels, self.u):
            label = transformation(label)
            if label is not None:
                i = labels.add(label)
                if i == len(rows):
                    rows.append(np.copy(row))
                else:  # If i > len(rows), something has gone terribly awry
                    rows[i] += row
        u, s = eigenmath.redecompose(np.vstack(rows), self.sigma)
        return self.__class__(u, s, labels)

    def merge_similar(self, others, weights=None):
        '''
        Construct a new AssocSpace formed by merging this one with another,
        assuming that the spaces were built from subsamples of the same set
        of data.
        '''
        if weights is None:
            weights = (1.0,) * (len(others) + 1)
        merged_labels, index_lists = self.labels.merge_many(
            [other.labels for other in others]
        )
        self_expanded = np.zeros((len(merged_labels), self.k))
        self_expanded[:self.u.shape[0], :] = self.u

        decompositions = [(self_expanded, self.sigma * weights[0])]

        for other, indices, weight in zip(others, index_lists, weights[:1]):
            other_expanded = np.zeros((len(merged_labels), other.k))
            other_expanded[indices, :] = other.u * weight
            decompositions.append((other_expanded, other.sigma * weight))

        new_u, new_sigma = eigenmath.combine_similar_eigenspaces(
            decompositions, self.k
        )

    def merge_dissimilar(self, other, weights=(1.0, 1.0)):
        '''
        Construct a new AssocSpace formed by merging this one with another.
        The two matrices can come from different sets of data.
        The result approximates the sum of the underlying associations from
        the two spaces, using the dimensionality of this space.

        If the two matrices contain entries sampled randomly from a more
        complete matrix, consider using `merge_similar`, which is simpler.

        By default the largest eigenvalue of each space is normalized to 1.0
        before merging.  This normalization can be changed by passing the
        weights argument, which is a tuple in the order (self, other).
        '''
        # Merge the labels and create expanded, aligned U matrices
        merged_labels, indices = self.labels.merge(other.labels)
        self_expanded = np.zeros((len(merged_labels), self.k))
        self_expanded[:self.u.shape[0], :] = self.u
        other_expanded = np.zeros((len(merged_labels), other.k))
        other_expanded[indices, :] = other.u

        self_weight, other_weight = weights

        # The largest eigenvalue is already normalized to 1 by the constructor
        new_u, new_sigma = eigenmath.combine_dissimilar_eigenspaces(
            [(self_expanded, self.sigma * self_weight),
             (other_expanded, other.sigma * other_weight)],
            self.k
        )

        return self.__class__(new_u, new_sigma, merged_labels)

    def truncated_to(self, k):
        '''
        Returns a copy of the space truncated to k dimensions.
        '''
        return self.__class__(self.u[:, :k], self.sigma[:k], self.labels)

    def select_rows(self, rows):
        '''
        Returns a subset of the space using a given set of row indices.
        '''
        selected_labels = self.labels[rows]
        if not isinstance(selected_labels, LabelSet):
            raise ValueError("Indexing must select a subset, not a single row")
        return self.__class__(self.u[rows, :], self.sigma, self.labels[rows])
    __getitem__ = select_rows

    def normalize_out(self, vector):
        """
        Makes a new AssocSpace orthogonal to a given vector, by
        removing the component corresponding to that vector from every
        row of the space.
        """
        vector = eigenmath.normalize(vector)
        magnitudes = np.dot(self.u, vector)
        projections = np.outer(magnitudes, vector)
        new_u = self.u - projections
        return self.__class__(new_u, self.sigma, self.labels)

    def __eq__(self, other):
        return (np.all(self.u == other.u) and
                np.all(self.sigma == other.sigma) and
                np.all(self.assoc == other.assoc) and
                self.labels == other.labels)

    def __ne__(self, other):
        return not (self == other)
