from assoc_space import AssocSpace, LabelSet
import numpy as np
import pickle
from nose.tools import eq_, assert_raises
from lumi_science.eigenmath import norm, normalize

DBNAME = None
DB = None

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


def test_pickle_round_trip():
    """An AssocSpace survives a round-trip to pickle format and back."""
    assoc = AssocSpace.from_entries(ENTRIES, k=3)
    pickled = pickle.dumps(assoc)
    assoc2 = pickle.loads(pickled)
    eq_(assoc, assoc2)


def test_dir_round_trip():
    assoc = AssocSpace.from_entries(ENTRIES, k=3)
    assoc.save_dir('/tmp/assoc_test')
    assoc2 = AssocSpace.load_dir('/tmp/assoc_test')
    eq_(assoc, assoc2)


def test_assoc_constructor():
    # Make a nice, normal AssocSpace
    u = np.asarray([[0, 1, 0.6], [1, 0, 0.8]])
    sigma = np.asarray([0.5, 0.3, 0.2])
    labels = LabelSet(['A', 'B'])
    assoc = AssocSpace(u, sigma, labels)
    eq_(assoc.k, 3)
    assert 'assoc' not in assoc.__dict__

    # Test some error conditions
    with assert_raises(ValueError):
        AssocSpace(u, np.asarray([0.0, -0.2, -0.4]), labels)
    with assert_raises(ValueError):
        AssocSpace(u, np.asarray([0.6, 0.4]), labels)
    with assert_raises(ValueError):
        AssocSpace(u, np.asarray([0.6, 0.7, 0.2]), labels)

    # Test assoc hinting
    assoc_matrix = assoc.assoc.copy()
    assoc_hinted = AssocSpace(u, sigma, labels, assoc=assoc_matrix)
    assert np.allclose(assoc_hinted.row_named('A'), assoc.row_named('A'))


def test_from_entries_and_from_matrix():
    # Note: for convenience from_matrix() is tested here implicitly, rather
    # than in a separate test.

    # Reject outright a space with no entries and a space with insufficient
    # entries
    assert AssocSpace.from_entries([], k=1) is None
    assert AssocSpace.from_entries([(1, 'apple', 'red')], k=1) is None

    # Build with mostly-default parameters and check some simple properties
    assoc_default = AssocSpace.from_entries(ENTRIES, k=4)
    eq_(assoc_default.k, 4)
    eq_(assoc_default.sigma[0], 1.0)
    assert assoc_default.assoc_between_two_terms('apple', 'red') > 0.5
    assert assoc_default.assoc_between_two_terms('red', 'red') > 0.999
    assert assoc_default.assoc_between_two_terms('lemon', 'red') < 0.2

    # Build with strip_a0=False; in this case we have negative eigenvalues,
    # so we lose an eigenvalue from the middle to make room for a0
    assoc_no_strip = AssocSpace.from_entries(ENTRIES, k=4, strip_a0=False)
    eq_(assoc_no_strip.k, 4)
    assert np.allclose(assoc_no_strip.sigma[-1] / assoc_no_strip.sigma[1],
                       assoc_default.sigma[-1])
    assert (np.allclose(assoc_no_strip.u[:, 1], assoc_default.u[:, 0]) or
            np.allclose(assoc_no_strip.u[:, 1], -assoc_default.u[:, 0]))

    # Build with normalize_gm=False
    assoc_no_norm = AssocSpace.from_entries(ENTRIES, k=4, normalize_gm=False)
    eq_(assoc_no_norm.k, 4)


# Filter function for test_filter
def _filter(label):
    if label[-1] == 'e':
        return
    if label == 'yellow':
        return 'red'
    return label


def test_filter():
    # Build and filter an assoc space
    assoc = AssocSpace.from_entries(ENTRIES, k=5)
    filtered = assoc.filter(_filter)

    # Check simple properties of the filtered space
    eq_(filtered.k, 5)
    eq_(' '.join(filtered.labels), 'red green celery banana lemon')

    # Check that redecomposition happened
    assert np.allclose(norm(filtered.u[:, 1]), 1.0)

    # Redecomposition can be kind of weird, but this result is intuitive
    assert (assoc.assoc_between_two_terms('red', 'banana') <
            filtered.assoc_between_two_terms('red', 'banana') <
            assoc.assoc_between_two_terms('yellow', 'banana'))


MORE_ENTRIES = [
    (2, 'apple', 'red'),
    (2, 'orange', 'blue'),
    (4, 'apple', 'tasty'),
    (1, 'banana', 'tasty'),
    (3, 'apple', 'orange'),
    (1.5, 'banana', 'ferret'),
    (4, 'ferret', 'yellow'),
    (0.5, 'blue', 'yellow')
]


def test_merging():
    # The actual math of merging is tested separately in test_eigenmath; here
    # we just spot-verify that AssocSpace is using it reasonably

    # Generate test assoc spaces and merge them
    assoc1 = AssocSpace.from_entries(ENTRIES, k=4)
    assoc2 = AssocSpace.from_entries(MORE_ENTRIES, k=4)
    merged = assoc1.merge_dissimilar(assoc2)

    # Check some simple things
    eq_(merged.k, 4)
    eq_(' '.join(merged.labels),
        'apple red green celery orange banana yellow lemon blue tasty ferret')
    assert merged.assoc_between_two_terms('ferret', 'yellow') > 0.5
    assert (assoc2.assoc_between_two_terms('apple', 'red') <
            merged.assoc_between_two_terms('apple', 'red') <
            assoc1.assoc_between_two_terms('apple', 'red'))


def test_truncation():
    # Simple test of truncation
    assoc = AssocSpace.from_entries(ENTRIES, k=3)
    truncated = assoc.truncated_to(2)
    assert np.allclose(truncated.u, assoc.u[:, :2])
    assert np.allclose(truncated.sigma, assoc.sigma[:2])
    eq_(truncated.labels, assoc.labels)
    assert 0.999 < norm(truncated.assoc[0]) < 1.0


def test_vectorizing_and_similar_terms():
    # Simple test for vectorizing weighted terms
    assoc = AssocSpace.from_entries(ENTRIES, k=3)
    weighted_terms = [('apple', 5), ('banana', 22), ('not a term', 17)]
    apple = assoc.row_named('apple')
    banana = assoc.row_named('banana')
    vector = assoc.vector_from_terms(weighted_terms)

    # The similarity of 'apple' to itself is approximately 1
    assert abs(assoc.assoc_between_two_terms('apple', 'apple') - 1.0) < 1e-3

    # 'apple' and 'banana' are at least 10% less similar to each other than
    # to themselves
    assert assoc.assoc_between_two_terms('apple', 'banana') < 0.9

    # The vector is some linear combination of apple and banana. Test this
    # by subtracting out apple and banana components, so that there is nothing
    # left.
    norm_apple = normalize(apple)
    banana_perp_apple = normalize(banana - norm_apple * norm_apple.dot(banana))
    residual = vector - norm_apple * norm_apple.dot(vector)
    residual -= banana_perp_apple * banana_perp_apple.dot(residual)
    assert norm(residual) < 1e-3

    # Simple test for finding similar terms
    labels, scores = zip(*assoc.terms_similar_to_vector(vector))
    eq_(list(scores), sorted(scores, reverse=True))
    assert labels.index('banana') < labels.index('apple')
    assert labels.index('apple') < labels.index('green')
    assert labels.index('apple') < labels.index('celery')
