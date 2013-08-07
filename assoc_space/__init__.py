import numpy as np
import scipy.sparse.linalg.eigen
from scipy.sparse import coo_matrix, spdiags
from ordered_set import OrderedSet
from collections import defaultdict
import os
import codecs

SMALL = 1e-6


def norm(vec, offset=None):
    if offset is None:
        offset = SMALL
    return offset + np.sqrt(np.sum(vec * vec))


def normalize(vec, offset=None):
    if offset is None:
        offset = SMALL
    return vec / norm(vec, offset)


def normalize_rows(dmat, offset=None):
    if offset is None:
        offset = SMALL
    squared = dmat * dmat
    sums = np.sqrt(squared.sum(1)) + offset
    return dmat / sums[:, np.newaxis]


class SparseEntryStorage(object):
    """
    Temporarily stores entries of a labeled sparse matrix in an efficient
    format.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets this SparseEntryStorage to being empty.
        """
        self.labels = OrderedSet()
        self.entries = defaultdict(float)

    def add_entry(self, entry):
        """
        Add a single triple of the form (value, row_label, col_label).
        """
        value, row_label, col_label = entry
        key = (self.labels.add(row_label), self.labels.add(col_label))
        self.entries[key] += value

    def add_entries(self, entries):
        """
        Add triples of the form (value, row_label, col_label).
        """
        for value, row_label, col_label in entries:
            key = (self.labels.add(row_label), self.labels.add(col_label))
            self.entries[key] += value

    def labels_and_matrix(self):
        """
        Return the labels and symmetrized sparse matrix.
        """
        # Borrowed from scipy.sparse.dok_matrix.tocoo()
        data = np.asarray(self.entries.values(), dtype='d')
        indices = np.asarray(self.entries.keys(), dtype=np.intc).T
        labels = self.labels

        matrix = coo_matrix((data, indices), shape=(len(labels), len(labels)))
        return labels, matrix + matrix.T


class AssocSpace(object):
    """
    An AssocSpace projects terms into a vector space shaped by their
    associations.

    AssocSpaces are intended to be treated as immutable. All mutation
    operations return a new AssocSpace instead. However, the new
    AssocSpace may share data, either as the same objects or as NumPy
    views, with the original AssocSpace. So avoid modifying matrices
    that have been used as inputs to an AssocSpace that you still
    need.
    """
    def __init__(self, u, sigma, labels):
        self.u = np.asarray(u, '>f4')
        self.sigma = np.asarray(sigma, '>f4')
        if not isinstance(labels, OrderedSet):
            labels = OrderedSet(labels)
        self.labels = labels
        assert len(self.labels) == self.u.shape[0], \
            '%s != %s' % (len(self.labels), self.u.shape[0])
        self.k = len(self.sigma)
        assert self.k == self.u.shape[1], \
            '%s != %s' % (self.k, self.u.shape[1])

        if self.sigma[0] == 0:
            raise ValueError('Cannot create AssocSpace with all-zero sigma.')

        self.assoc = normalize_rows(
            self.spectrally_associate(self.u),
            offset=0.0001
        )

    @classmethod
    def from_matrix(cls, mat, k, labels, strip_a0=False):
        '''
        Creates an AssocSpace with dimensionality `k` from a matrix of
        new associations `mat`.
        '''
        u, s = eigensystem(mat, k=k, strip_a0=strip_a0)
        return cls(u, s, labels)

    @classmethod
    def from_entries(cls, entries, k, **kwargs):
        """
        Build an AssocSpace out of its sparse labeled entries, as triples
        in (value, row_label, col_label) form.

        strip_a0 will remove the first singular component, which is often
        uninformative.

        normalize_gm will divide each entry by the geometric mean of the sum of
        the column and the sum of the row.  This is one iteration of a process
        that might eventually normalize the rows and columns to have unit sum.
        However, we want to suppress sufficiently rare terms altogether, so we
        add an offset.  Ugly, but works well enough.

        At Luminoso, we have found that k=150 and offset_weight=8e-6 are
        parameters that work well for various kinds of input data. Your
        mileage may vary.
        """
        storage = SparseEntryStorage()
        storage.add_entries(entries)
        return cls.from_sparse_storage(storage, k, **kwargs)

    @classmethod
    def from_sparse_storage(cls, storage, k, strip_a0=False,
                            offset_weight=8e-6,
                            normalize_gm=True):
        """
        Build an AssocSpace from a SparseEntryStorage.

        This is a helper method; see from_entries() for usage and a
        description of the parameters.
        """
        labels, matrix = storage.labels_and_matrix()
        if normalize_gm:
            sums = np.absolute(matrix).sum(0)
            offset = np.sum(sums) * offset_weight
            normalizer = spdiags(1.0 / np.sqrt(sums + offset), 0,
                                 len(labels), len(labels))
            matrix = normalizer * matrix * normalizer
        return cls.from_matrix(matrix, k, labels, strip_a0=strip_a0)

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
        if not isinstance(selected_labels, OrderedSet):
            raise ValueError("Indexing must select a subset, not a single row")
        return self.__class__(self.u[rows, :], self.sigma, self.labels[rows])
    __getitem__ = select_rows

    def select_averaged_rows(self, row_dict):
        """
        Given a mapping from labels to row-indices, returns a space in which
        the row with a given label is the average of those row-indices.
        """
        labels = OrderedSet()
        new_u = np.zeros((len(row_dict), self.k))
        for label, indices in row_dict.items():
            rownum = labels.add(label)
            old_rows = self.u[indices, :]
            new_u[rownum] = sum(old_rows) / len(old_rows)
        return self.__class__(new_u, self.sigma, labels)

    def complex_filter_by_label(self, filter_func, transform_func):
        """
        Like filter_by_label, but requires a `transform_func` because it
        averages rows for labels with identical transform_func results, rather
        than taking one row arbitrarily.
        """
        labels = defaultdict(list)
        for i in xrange(len(self.labels)):
            if filter_func(self.labels[i]):
                labels[transform_func(self.labels[i])].append(i)
        return self.select_averaged_rows(labels)

    def filter_by_label(self, filter_func, transform_func=None):
        '''
        Return the subset of the space whose labels return true in
        `filter_func`. If `transform_func` is provided, it will additionally
        ensure that the labels are unique when run through that function.
        '''
        indices = []
        used = set()
        if transform_func is None:
            transform_func = lambda x: x
        for i in xrange(len(self.labels)):
            if filter_func(self.labels[i]):
                transformed = transform_func(self.labels[i])
                if transformed not in used:
                    indices.append(i)
                    used.add(transformed)
        return self.select_rows(indices)

    def save_dir(self, dirname):
        """
        Save the contents of an AssocSpace to a directory on disk.
        """
        dirname = dirname.rstrip('/')
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        np.save(dirname + '/u.npy', self.u)
        np.save(dirname + '/sigma.npy', self.sigma)
        label_file = codecs.open(dirname + '/labels.txt', 'w',
                                 encoding='utf-8')
        for label in self.labels:
            print >> label_file, label
        label_file.close()

    @classmethod
    def load_dir(cls, dirname):
        """
        Load an AssocSpace from a directory on disk. The returned object
        will be an instance of the class that you called .load_dir on.
        """
        dirname = dirname.rstrip('/')
        u = np.load(dirname + '/u.npy', mmap_mode='r')

        sigma = np.load(dirname + '/sigma.npy')
        label_file = codecs.open(dirname + '/labels.txt',
                                 'r', encoding='utf-8')
        labels = OrderedSet()
        for label in label_file:
            labels.append(label.rstrip('\n'))
        return cls(u, sigma, labels)

    def subtract_out(self, vector):
        """
        Makes a new AssocSpace orthogonal to a given vector, by
        subtracting out the component corresponding to that vector from every
        row of the space.
        """
        vector = normalize(vector)
        magnitudes = np.dot(self.u, vector)
        projections = np.outer(magnitudes, vector)
        new_u = self.u - projections
        return self.__class__(new_u, self.sigma, self.labels)

    def expanded_to(self, rows, cols, labels=None):
        '''
        Returns a copy of the space expanded to (rows, cols). Does not
        change the dimensionality of the space. New rows and columns
        project to zero. An error is raised if the existing data does
        not fit.

        If `labels` is passed, it is used as the row and column
        labels.
        '''
        if rows < self.u.shape[0]:
            raise ValueError("Number of rows is insufficient to fit existing "
                             "data.")
        if cols < self.u.shape[1]:
            raise ValueError("Number of columns is insufficient to fit "
                             "existing data.")
        k = self.k

        if labels is None:
            labels = self.labels

        newU = np.zeros((rows, k))
        newU[:self.u.shape[0], :] = self.u

        return self.__class__(newU, self.sigma, labels)

    def with_first_axis_dropped(self):
        return self.__class__(
            self.u[:, 1:],
            self.sigma[1:],
            self.labels)

    def spectrally_associate(self, vec):
        """
        Multiply vec by e^(S/2), where S is a rescaled version of Sigma.
        This operation can also be broadcast across an entire U matrix.

        This is a cool trick that simulates spreading activation on its
        input vectors.
        """
        # S/2, you say?  Why?
        # The short version: when we generate vectors for two things and dot
        # them together, both of them will have an e^(S/2) in them, meaning
        # that between them they have the full e^S.
        return np.multiply(vec, np.exp(self.sigma / 2 / np.max(self.sigma)))

    def all_pairs_similarity(self):
        u_sigma = np.multiply(self.u, self.sigma / np.max(self.sigma))
        return u_sigma.dot(u_sigma.T)

    def all_pairs_association(self):
        return self.assoc.dot(self.assoc.T)

    def vector_from_terms(self, terms):
        """
        Get a category vector representing the given set of weighted terms,
        expressed as (term, weight) tuples.
        """
        outvec = np.zeros((self.k,))

        # If none of the terms are real terms, return zero vector.
        if not any([term in self.labels for (term, weight) in terms]):
            return outvec

        # Normalize weights to have max 1.0, so that (hopefully) it won't
        # end up with any really tiny numbers.
        max_weight = max([abs(weight) for (term, weight) in terms
                          if term in self.labels])
        reweighted_terms = [(term, weight / max_weight)
                            for (term, weight) in terms if term in self.labels]
        total_sq_weight = sum([weight ** 2
                              for (term, weight) in reweighted_terms])

        norm = total_sq_weight ** .5
        if norm <= 0.0:
            # This should never happen.
            return outvec
        for term, weight in reweighted_terms:
            outvec += self.row_named(term) * weight / norm

        return outvec

    def row_named(self, term):
        return self.assoc[self.labels.index(term)]

    def assoc_between_two_terms(self, term1, term2):
        return np.dot(self.row_named(term1), self.row_named(term2))

    def terms_similar_to_vector(self, vec):
        """
        Take in a category vector, and returns a list of (term, similarity)
        tuples.  It will be sorted in descending order by similarity.
        """
        similarity = zip(self.labels, np.dot(self.assoc, vec))
        similarity.sort(key=lambda item: -item[1])
        return similarity

    def __eq__(self, other):
        return (np.all(self.u == other.u) and
                np.all(self.sigma == other.sigma) and
                np.all(self.assoc == other.assoc) and
                self.labels == other.labels)

    def __ne__(self, other):
        return not (self == other)


def eigensystem(mat, k, strip_a0=False):
    """
    Find the eigenvalues and eigenvectors of the given symmetric sparse matrix.

    This is mostly a wrapper around SciPy's eigsh() function, except we:
     * Convert to Compressed Sparse Rows for speed
     * Sort by descending eigenvalue
     * Trim or pad with zeros to the desired rank
     * Optionally remove the largest eigenvalue

    k is the desired rank, and strip_a0=True removes the largest eigenvalue.
    """
    # Compute the actual number of eigenvalues to find.
    # It can't actually solve for all of them.
    offset = 1 if strip_a0 else 0
    real_k = min(k + offset, mat.shape[0] - 1)
    if real_k < 1:
        raise ValueError("Attempted to solve for no eigenvalues.")

    # Find the largest eigenvalues. 'LA' means 'largest algebraic': that is,
    # we don't want large negative eigenvalues.
    S, U = scipy.sparse.linalg.eigen.eigsh(mat.tocsr(), k=real_k, which='LA')

    # Sort and trim
    order = np.argsort(S)[::-1]
    S = S[order][offset:]
    U = U[:, order][:, offset:]

    # Pad.  When we have a more recent NumPy we should use the pad() function
    to_pad = k - U.shape[1]
    if to_pad > 0:
        S = np.append(S, np.zeros((to_pad,)), axis=0)
        U = np.append(U, np.zeros((U.shape[0], to_pad)), axis=1)

    return U, S
