from assoc_space import AssocSpace
import numpy as np
import pickle
from nose.tools import eq_
from nose.plugins.attrib import attr

DBNAME = None
DB = None

entries = [(2, 'apple', 'red'),
 (2, 'orange', 'orange'),
 (1, 'apple', 'green'),
 (1, 'celery', 'green'),
 (3, 'apple', 'orange'),
 (3, 'banana', 'orange'),
 (1, 'banana', 'yellow'),
 (0.5, 'lemon', 'yellow')]

# Entries to simulate 5.1.1 and 5.1.3 background spaces, to ensure that labels
# are filtered properly. (Note that a good stemmer should turn "apples" and
# "manzanas" into "apple" and "manzana" if it runs on the labels. A bad stemmer
# will turn "manzanas" into "manzan", as in the test below.)
cn_511_entries = [(1, one, two) for one, two in
                  [('/c/en/apples', '/c/es/manzanas'),
                   ('/c/en/orange/n/fruit', '/c/es/naranja/n/fruta'),
                   ('/c/en/pear', '/c/es/pera'),
                   ('/c/en/banana', '/c/es/banana'),
                   ('/c/en/pear', '/c/es/banana'),
                   ('/c/en/apples', '/c/es/pera'),
                   ]
                  ]

cn_513_entries = [(1, one, two) for one, two in
                  [('/c/en/apples', '/c/es/manzanas'),
                   ('/c/en/orange/neg', '/c/es/naranja/neg'),
                   ('/c/en/pear', '/c/es/pera'),
                   ('/c/en/banana', '/c/es/banana'),
                   ('/c/en/pear', '/c/es/banana'),
                   ('/c/en/apples', '/c/es/pera'),
                   ]
                  ]

@attr(priority=1)
def test_strip_a0():
    """When stripping a0, AssocSpace uses axes [1,k] instead of [0,k-1]."""
    assoc = AssocSpace.from_entries(entries, 3, strip_a0=False)
    assoc_stripped_mat = AssocSpace.from_entries(entries, 3, strip_a0=True)

    # Check for the same number of k
    eq_(assoc.u.shape[1], 3)
    assert np.allclose(np.abs(assoc.u[:,1]), np.abs(assoc_stripped_mat.u[:,0]))

    # check that the ratio between sigma values is preserved
    assert np.allclose(assoc.sigma[1] / assoc.sigma[2],
            assoc_stripped_mat.sigma[0] / assoc_stripped_mat.sigma[1])

    assoc_stripped_dropa0 = AssocSpace.from_entries(entries, 3).with_first_axis_dropped()
    assert np.allclose(np.abs(assoc.u[:,1]),
            np.abs(assoc_stripped_dropa0.u[:,0]))
    assert np.allclose(assoc.sigma[1] / assoc.sigma[2],
            assoc_stripped_dropa0.sigma[0] / assoc_stripped_dropa0.sigma[1])

@attr(priority=1)
def test_pickle_round_trip():
    """An AssocSpace survives a round-trip to pickle format and back."""
    assoc = AssocSpace.from_entries(entries, 3)
    pickled = pickle.dumps(assoc)
    assoc2 = pickle.loads(pickled)
    eq_(assoc, assoc2)

@attr(priority=1)
def test_dir_round_trip():
    assoc = AssocSpace.from_entries(entries, 3)
    assoc.save_dir('/tmp/assoc_test')
    assoc2 = AssocSpace.load_dir('/tmp/assoc_test')
    eq_(assoc, assoc2)

# TODO: lots more like this
@attr(priority=1)
def test_association_calculations():
    assoc = AssocSpace.from_entries(entries, 3)
    assert abs(assoc.assoc_between_two_terms('apple', 'apple') - 1.0) < 1e-3
    assert assoc.assoc_between_two_terms('apple', 'banana') < 0.9
