from nose.tools import eq_
import numpy as np
import scipy.sparse

from assoc_space.eigenmath import norm, normalize, normalize_rows, \
    eigensystem, redecompose, combine_eigenspaces


def compare_cols_within_sign(m1, m2):
    '''
    Compare the columns of m1 and m2, allowing them to differ by sign.
    '''
    eq_(m1.shape, m2.shape)
    for i in range(m1.shape[1]):
        col1 = m1[:, i]
        col2 = m2[:, i]
        assert np.allclose(col1, col2) or np.allclose(col1, -1 * col2)


def undecompose(u, s):
    '''
    Given U and S, return the represented matrix U S U^T.
    '''
    return (u * s).dot(u.T)


def test_norm_and_normalize():
    vec = np.asarray([8.0, 9.0, 12.0])
    assert np.allclose(norm(vec), 17.0)
    assert np.allclose(normalize(vec), vec / 17.0)
    # We normalize the zero vector to itself rather than raising an error
    assert (np.zeros(5) == normalize(np.zeros(5))).all()


def test_normalize_rows():
    arr = np.asarray([[3.0, 4.0], [0.3, 0.4]])
    assert np.allclose(normalize_rows(arr), [[0.6, 0.8], [0.6, 0.8]])
    normalized_with_offset = normalize_rows(arr, offset=0.001)
    assert (normalized_with_offset[0] > normalized_with_offset[1]).all()


# The matrix here is the symmetric Markov matrix
#     0.0 0.3 0.6 0.1
#     0.3 0.0 0.1 0.6
#     0.6 0.1 0.0 0.3
#     0.1 0.6 0.3 0.0
# with eigenvalues (1, 0.2, -0.4, -0.8) and eigenvectors (up to sign)
#     [0.5, 0.5, 0.5, 0.5]
#     [-0.5, 0.5, -0.5, 0.5]
#     [0.5, 0.5, -0.5, -0.5]
#     [0.5, -0.5, -0.5, 0.5]


def test_eigensystem():
    spmat = scipy.sparse.coo_matrix(
        ([0.3, 0.6, 0.1, 0.3, 0.1, 0.6, 0.6, 0.1, 0.3, 0.1, 0.6, 0.3],
         ([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
          [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2])),
        shape=(4, 4))

    # Eigenvalues are as expected; note that we get the largest magnitudes,
    # but in algebraic order
    u1, s1 = eigensystem(spmat, 3, strip_a0=False)
    assert np.allclose(s1, [1, -0.4, -0.8])
    # Eigenvectors are eigenvectors
    for i, si in enumerate(s1):
        ui = u1[:, i]
        assert np.allclose(spmat.dot(ui), si * ui)
    # Eigenvectors are orthonormal
    assert np.allclose(u1.T.dot(u1), np.identity(s1.shape[0]))

    # Stripping a0 removes the right eigenvalue
    u2, s2 = eigensystem(spmat, 2, strip_a0=True)
    assert np.allclose(s2, s1[1:])
    compare_cols_within_sign(u2, u1[:, 1:])

    # Asking for way too many eigenvalues is okay
    u3, s3 = eigensystem(spmat, 5)
    assert np.allclose(s3, s1)
    compare_cols_within_sign(u3, u1)


def test_combine_dissimilar_eigenspaces():
    # This takes two rank 2 decompositions of different matrices and generates
    # one rank 3 decomposition.  That rank 3 decomposition should be the top
    # 3 dimensions of the decomposition of the sum of the original matrices.

    # The decompositions
    u1 = np.asarray([[0.5, 0.5],
                     [0.5, -0.5],
                     [-0.5, -0.5],
                     [-0.5, 0.5],
                     [0.0, 0.0]])
    s1 = np.asarray([1.0, 0.7])
    u2 = np.asarray([[0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [-0.8, 0.6],
                     [0.6, 0.8]])
    s2 = np.asarray([1.0, 0.5])

    # The combined decomposition
    u_c, s_c = combine_eigenspaces(u1, s1, u2, s2, 3)

    # The decomposition of the sum of the matrices
    s_ref, u_ref = np.linalg.eigh(undecompose(u1, s1) + undecompose(u2, s2))

    # Sort the reference decomposition in decreasing algebraic order, to match
    # the sorting combine_eigenspaces() uses
    order = np.argsort(s_ref)[::-1]
    s_ref = s_ref[order]
    u_ref = u_ref[:, order]

    # Check the output
    assert np.allclose(s_c, s_ref[:3])
    compare_cols_within_sign(u_c, u_ref[:, :3])


def test_redecompose():
    # This is matrix #1 from the previous test, but with a row missing.
    u_in = np.asarray([[0.5, 0.5],
                       [0.5, -0.5],
                       [-0.5, 0.5],
                       [0.0, 0.0]])
    s_in = np.asarray([1.0, 0.7])

    # Redecompose
    u_out, s_out = redecompose(u_in, s_in)

    # The represented matrix should be unchanged...
    assert np.allclose(undecompose(u_in, s_in), undecompose(u_out, s_out))

    # ...but the output should be a proper eigenvector decomposition
    assert np.allclose(u_out.T.dot(u_out), np.identity(2))
