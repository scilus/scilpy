

def is_float(value):
    """Returns True if the argument can be casted to a float."""
    try:
        float(value)
        return True
    except ValueError:
        return False


def recursive_update(d, u, from_existing=False):
    """Harmonize a dictionary to garantee all keys exists at all sub-levels."""
    import collections.abc

    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            if k not in d and from_existing:
                d[k] = u[k]
            else:
                d[k] = recursive_update(d.get(k, {}), v,
                                        from_existing=from_existing)
        else:
            if not from_existing:
                d[k] = float('nan')
            elif k not in d:
                d[k] = float('nan')
    return d


def recursive_print(data):
    """Print the keys of all layers. Dictionary must be harmonized first."""
    import collections.abc

    if isinstance(data, collections.abc.Mapping):
        print(list(data.keys()))
        recursive_print(data[list(data.keys())[0]])
    else:
        return
