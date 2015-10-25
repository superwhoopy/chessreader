def singleton_get(singleton_set: set):
    '''Return the single object in a set

    Raises:
        IndexError: if `singleton_set` is not a singleton
    '''
    if len(singleton_set) != 1:
        raise IndexError()
    return next(iter(singleton_set))
