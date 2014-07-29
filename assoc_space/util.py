'''
Utilities.
'''


class lazy_property(object):
    """
    Decorator for properties that you want to compute, lazily, just once.

    From http://stackoverflow.com/a/6849299 and similar code in simplenlp.
    See also http://docs.python.org/2/howto/descriptor.html for reference.
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        result = self.func(instance)
        setattr(instance, self.func.__name__, result)
        return result
