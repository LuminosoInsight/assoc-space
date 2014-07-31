from assoc_space import SparseEntryStorage
from nose.tools import eq_


ENTRIES = [
    (4, 'apple', 'red'),
    (1, 'apple', 'green'),
    (1, 'celery', 'green'),
    (3, 'apple', 'orange'),
    (3, 'banana', 'orange'),
    (1, 'banana', 'yellow'),
    (0.5, 'lemon', 'yellow'),
    (1.5, 'orange', 'lemon'),
    (0.1, 'apple', 'lemon'),
    (0.2, 'banana', 'lemon')
]


def test_sparse_storage():
    # Simple tests for SparseEntryStorage.
    bucket = SparseEntryStorage()

    # Getting labels and matrix from an empty storage bucket does not crash
    matrix, labels, matrix_sum = bucket.get_matrix_and_metadata()
    eq_(len(labels), 0)
    eq_(matrix.shape, (0, 0))
    eq_(matrix_sum, 0.0)

    # Actually add some things and check again
    bucket.add_entries(ENTRIES)
    matrix, labels, matrix_sum = bucket.get_matrix_and_metadata()
    eq_(' '.join(labels), 'apple red green celery orange banana yellow lemon')
    eq_(matrix[0, 1], 4)
    eq_(matrix[6, 5], 1)
    eq_(matrix[4, 2], 0)
    assert abs(matrix_sum - 15.3) < .000001
