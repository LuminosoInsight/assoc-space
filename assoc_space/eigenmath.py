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


def combine_eigenspaces(U_X, S_X, U_Y, S_Y, rank):
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
    Sp = Sp[order][:rank]
    Up = Up[:, order][:, :rank]

    # Done!
    return np.c_[U_X, Q].dot(Up), Sp         # Eqn. (4)


def combine_multiple_eigenspaces(US_list, rank=None):
    """
    Given a list of eigenvalue decompositions of a list of matrices
    [X_0, X_1, ..., X_n], find the decomposition of their sum
    X_0 + X_1 + ... + X_n .

    The decompositions must have aligned labels; that is, row r of matrix X_i
    should refer to the same thing as row r of matrix X_j, even if that means
    the row has to be the zero vector. The `AssocSpace.merged_with`
    function is a higher-level version that takes care of row alignment.

    Inputs:

    - US_list: a list of tuples (U_i, S_i) that are the decomposition of X_i
    - rank: the number of dimensions to trim the result to. By default, no
        trimming occurs.

    Returns: the new decomposition U, S.

    The algorithm is adapted from Brand 2006 (MERL TR2006-059) [1], section 2,
    to operate on eigenvalue decompositions instead of SVDs.

    [1] http://www.merl.com/publications/docs/TR2006-059.pdf
    """

    # These are used mainly for initializing matrices of the right dimension
    # n is the number of decompositions.
    n = len(US_list)
    # l is the "long" side of our decomposition. We check to make sure that 
    # the matrices all have the same long dimension, otherwise their labels
    # are misaligned.
    l_list = [U.shape[0] for U, S in US_list]
    assert len(l_list) == len([l for l in l_list if l == l_list[0]])
    l = l_list[0]
    # k is our "thin" dimension, usually 150. This may differ from U_i to U_j.
    k_list = [len(S) for U, S in US_list]
    dim = sum(k_list)
    if rank is None:
        rank = dim

    # Check to make sure that the columns of U_0 are orthonormal. If not,
    # normalize using QR decomposition. This behavior replaces the redecompose
    # function. We do this check for speed; the QR decomposition will work
    # even if the columns are orthonormal.
    U_0 = US_list[0][0]
    I = np.identity(k_list[0])
    if np.allclose(U_0.T.dot(U_0), I):
        QR_list = [(U_0, I)]
    else:
        QR_list = [np.linalg.qr(U_0)]

    # Create the basis Q_0, ..., Q_{n-1}, as well as the appropriate R_i.
    # Each Q_i depends on the sum of Q_j * Q_j^T * U_i for all j < i.
    # We keep track of this in M_sum. There are some performance savings to
    # be realized here, since we will immediately recompute Q_j^T * U_i below.
    # However, the only way I could think of to store those is in a four-
    # dimensional array, and that may be a task for another time.
    for i, (U, S) in enumerate(US_list):
        # We've already taken care of Q_0, but this is the easiest way to
        # prevent off-by-one errors.
        if i == 0:
            continue
        M_sum = np.zeros((l, k_list[i]))
        for (Q, R) in QR_list:
            M_sum += Q.dot(Q.T.dot(U))
        Q, R = np.linalg.qr(U - M_sum)
        QR_list.append((Q, R))

    # Construct components of each U in the basis Q_0, ..., Q_{n-1}, and use
    # them to express the sum of the X_i in that basis.
    K = np.zeros((dim, dim))
    for i, (U, S) in enumerate(US_list):
        V_list = []
        for j, (Q, R) in enumerate(QR_list):
            if j < i:
                V_list.append(Q.T.dot(U))
            elif j == i:
                V_list.append(R)
            else:
                V_list.append(np.zeros((k_list[j], k_list[i])))
        V = np.concatenate(V_list)
        K += (V * S).dot(V.T)

    # Diagonalize
    Sp, Up = np.linalg.eigh(K)

    # Sort and trim - we do this on the small matrices, for speed
    order = np.argsort(Sp)[::-1]
    Sp = Sp[order][:rank]
    Up = Up[:, order][:, :rank]

    # Done!
    return np.concatenate([Q for Q, R in QR_list], axis=1).dot(Up), Sp


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
