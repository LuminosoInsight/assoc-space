from assoc_space import LabelSet
from nose.tools import eq_, assert_raises


def test_label_set():
    # Creating a LabelSet from an existing list
    s = LabelSet(['a', 'b', 'e'])

    # add()
    s.add('d')
    s.add('c')
    s.add('d')
    eq_(len(s), 5)
    eq_(s[3], 'd')
    eq_(s[4], 'c')

    # index, __contains__, __getitem__, __iter__
    eq_(s.index('b'), 1)
    assert_raises(KeyError, s.index, 'f')
    assert 'b' in s
    assert 'q' not in s
    eq_(s[2], 'e')
    with assert_raises(IndexError):
        s[6]
    eq_(list(s), ['a', 'b', 'e', 'd', 'c'])

    # Copying generates a distinct LabelSet; __eq__ works
    s_copy = s.copy()
    eq_(s, s_copy)
    s_copy.add('f')
    eq_(len(s.items), 5)
    eq_(len(s.indices), 5)

    # merged
    s1 = LabelSet(['f', 'b', 'g', 'c'])
    merged, indices = s.merge(s1)
    eq_(len(s), 5)
    eq_(merged, LabelSet(['a', 'b', 'e', 'd', 'c', 'f', 'g']))
    eq_(indices, [5, 1, 6, 4])
