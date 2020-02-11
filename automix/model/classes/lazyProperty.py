import functools

class lazy_property(object):
    '''
    meant to be used for lazy evaluation of an object attribute.
    property should represent non-mutable data, as it replaces itself.
    from https://stackoverflow.com/questions/3012421/python-memoising-deferred-lookup-property-decorator
    '''

    def __init__(self, fget):
        self.fget = fget

        # copy the getter function's docstring and other attributes
        functools.update_wrapper(self, fget)

    def __get__(self, obj, cls):
        if obj is None:
            return self

        value = self.fget(obj)
        setattr(obj, self.fget.__name__, value)
        return value
