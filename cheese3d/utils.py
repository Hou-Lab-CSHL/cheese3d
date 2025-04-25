import itertools

def unzip(iter):
    return tuple(list(x) for x in zip(*iter))

def maybe(this, that):
    return that if this is None else this
