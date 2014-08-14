'''
Compatibility functions for python 2 vs. python 3.
'''

import sys

# Lines like "range = range" make the names importable from the outside
if sys.version_info.major == 2:
    import itertools
    range = xrange
    iteritems = lambda d: d.iteritems()
    values = lambda d: d.values()
    izip = itertools.izip
    zip = zip
    FileNotFoundError = IOError
    basestring = basestring
else:
    range = range
    iteritems = lambda d: d.items()
    values = lambda d: list(d.values())
    izip = zip
    zip = lambda *x: list(izip(*x))
    FileNotFoundError = FileNotFoundError
    basestring = str
