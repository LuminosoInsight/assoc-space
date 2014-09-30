import math
import numpy as np
import scipy.sparse.linalg.eigen


def norm(vec):
    """
    Find the norm of the given (one-dimensional) vector.
    """
    return math.sqrt(vec.dot(vec))


def normalize(vec):
    """
    Normalize the given vector.  Zero vectors remain zero.
    """
    vec_norm = norm(vec)
    if vec_norm == 0.0:
        return vec
    return vec / vec_norm


def row_norms(mat):
    return np.sqrt((mat * mat).sum(axis=1))


def col_norms(mat):
    return np.sqrt((mat * mat).sum(axis=0))


def normalize_rows(mat, offset=0.0):
    """
    Normalize the rows of the given matrix.

    If desired, pass an offset which will be added to the row norms, to cause
    very small rows to stay small in a smooth fashion.
    """
    return mat / (row_norms(mat) + offset)[:, np.newaxis]


def eigensystem(mat, k, strip_a0=False):
    """
    Find the eigenvalues and eigenvectors of the given symmetric sparse matrix.

    This is mostly a wrapper around SciPy's eigsh() function, except we:
      * Convert to Compressed Sparse Rows for speed
      * Sort by descending eigenvalue
      * Optionally remove the largest eigenvalue

    k is the desired rank, and strip_a0=True removes the largest eigenvalue.
    """
    # Compute the actual number of eigenvalues to find.
    # It can't actually solve for all of them.
    offset = 1 if strip_a0 else 0
    real_k = min(k + offset, mat.shape[0] - 1)
    if real_k < 1:
        raise ValueError("Attempted to solve for no eigenvalues.")

    # Find the eigenvalues
    S, U = scipy.sparse.linalg.eigen.eigsh(mat.tocsr(), k=real_k)

    # Sort and trim
    order = np.argsort(S)[::-1]
    S = S[order][offset:]
    U = U[:, order][:, offset:]

    return U, S


def combine_eigenspaces(U_X, S_X, U_Y, S_Y, rank=None):
    '''
    Given the eigenvalue decompositions of X and Y, find that of (X + Y).

    The decompositions must have aligned labels; that is, row i of matrix X
    should refer to the same thing as row i of matrix Y, even if that means
    the row has to be the zero vector. The `AssocSpace.merged_with`
    function is a higher-level version that takes care of row alignment.

    Inputs:

    - U_X, S_X: the decomposition of matrix X.
    - U_Y, S_Y: the decomposition of matrix Y.
    - rank: the number of dimensions to trim the result to.

    Returns: the new decomposition U, S.

    The algorithm is adapted from Brand 2006 (MERL TR2006-059) [1], section 2,
    to operate on eigenvalue decompositions instead of SVDs.

    [1] http://www.merl.com/publications/docs/TR2006-059.pdf
    '''
    # Find the basis for the orthogonal component of U_Y
    M_1 = U_X.T.dot(U_Y)
    Q, R = np.linalg.qr(U_Y - U_X.dot(M_1))  # Eqn. (1)

    # Express X + Y in the combined basis
    M_2 = np.r_[M_1, R]
    K = (np.asarray(M_2) * S_Y).dot(M_2.T)   # Eqn. (2), right side of sum
    for i in range(len(S_X)):
        K[i, i] += S_X[i]                    # Eqn. (2), left side of sum

    # Diagonalize
    Sp, Up = np.linalg.eigh(K)               # Eqn. (3)

    # Sort and trim - we do this on the small matrices, for speed
    order = np.argsort(Sp)[::-1]
    
    Sp = Sp[order]
    Up = Up[:, order]
    if rank is not None:
        Sp = Sp[:rank]
        Up = Up[:, :rank]

    # Done!
    return np.c_[U_X, Q].dot(Up), Sp         # Eqn. (4)


def redecompose(U, S):
    '''
    Given a "decomposition" U S U^T of a matrix X, find its eigenvalue
    decomposition.  U need not have normalized or orthogonal columns.

    This is useful if you have mangled a previous decomposition in some way
    and want to restore proper orthonormal columns with correct eigenvalues.
    '''
    # This is just a small version of the algorithm from combine_eigenspaces.
    # Find a basis for the space spanned by U
    Q, R = np.linalg.qr(U)

    # Express X in this basis and diagonalize
    Sp, Up = np.linalg.eigh((np.asarray(R) * S).dot(R.T))

    # Sort - we do this on the small matrices, for speed
    order = np.argsort(Sp)[::-1]
    Sp = Sp[order]
    Up = Up[:, order]

    # Done!
    return Q.dot(Up), Sp
