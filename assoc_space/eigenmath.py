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
    return np.sqrt((mat ** 2).sum(axis=1))


def col_norms(mat):
    return np.sqrt((mat ** 2).sum(axis=0))


def normalize_rows(mat, offset=0.0):
    """
    Normalize the rows of the given matrix.

    If desired, pass an offset which will be added to the row norms, to cause
    very small rows to stay small in a smooth fashion.
    """
    return mat / (np.sqrt((mat * mat).sum(1)) + offset)[:, np.newaxis]


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


def combine_similar_eigenspaces(decompositions, rank):
    '''
    Find the eigenvalue decomposition of a matrix, Z, given the
    decompositions of several random subsamples of it.

    The decompositions must have aligned labels; that is, row i of matrix X
    should refer to the same thing as row i of matrix Y, even if that means
    the row has to be the zero vector. The `AssocSpace.merge_similar` function
    is a higher-level version that takes care of row alignment.

    Inputs:

    - decompositions, a list of pairs (U_i, S_i) for each matrix to be
      combined.
    - rank, the number of dimensions to trim the result to.

    Returns: the new decomposition U, S.

    This algorithm works by simply averaging the (U * S) matrices and
    re-decomposing the result. It is a modification of El Karoui and
    d'Aspermont 2010, "Second order accurate distributed eigenvector
    computation for extremely large matrices". The modification accounts for
    the fact that we care about eigenvalues as well as eigenvectors.

    Averaging the eigenvectors requires that we standardize their arbitrary
    sign, so that equivalent eigenvectors don't end up subtracting from each
    other. To do this, we find the row with the largest magnitude across all
    the input matrices, and set that row as the positive direction for all
    eigenvectors.
    '''
    U_0, S_0 = decompositions[0]
    row_mags = row_norms(U_0)
    for U_i, _ in decompositions[1:]:
        row_mags *= row_norms(U_i)

    best_row = np.argmax(row_mags)

    scale_factors = S_0 * np.sign(U_0[best_row])
    combined_US = U_0 * scale_factors
    for U_i, S_i in decompositions[1:]:
        scale_factors = S_i * np.sign(U_i[best_row])
        combined_US += (U_i * scale_factors)

    S_tot = (combined_US ** 2).sum(axis=0) ** .5
    U_tot = combined_US / S_tot
    return redecompose(U_tot, S_tot)


def combine_dissimilar_eigenspaces(decompositions, rank):
    '''
    Given the eigenvalue decompositions of X and Y, find that of (X + Y).

    The decompositions must have aligned labels; that is, row i of matrix X
    should refer to the same thing as row i of matrix Y, even if that means
    the row has to be the zero vector. The `AssocSpace.merge_dissimilar`
    function is a higher-level version that takes care of row alignment.

    Inputs:

    - decompositions, which is a list of (U, S) for the two input
      matrices. That is, the list should contain [(U_X, S_X), (U_Y, S_Y)].
    - rank, the number of dimensions to trim the result to.

    Returns: the new decomposition U, S.

    This function signature is intended to be similar to
    combine_similar_eigenspaces, although that function can take any number of
    eigenspaces.

    The algorithm is adapted from Brand 2006 (MERL TR2006-059) [1], section 2,
    to operate on eigenvalue decompositions instead of SVDs.

    [1] http://www.merl.com/publications/docs/TR2006-059.pdf
    '''
    assert len(decompositions) == 2, "Can only combine two eigenspaces at a time"
    U_X, S_X = decompositions[0]
    U_Y, S_Y = decompositions[1]

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
    Sp = Sp[order][:rank]
    Up = Up[:, order][:, :rank]

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
