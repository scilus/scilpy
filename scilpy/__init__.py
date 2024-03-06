

def get_home():
    import os
    """ Set a user-writeable file-system location to put files. """
    if 'SCILPY_HOME' in os.environ:
        scilpy_home = os.environ['SCILPY_HOME']
    else:
        scilpy_home = os.path.join(os.path.expanduser('~'), '.scilpy')
    return scilpy_home


def get_root():
    import os
    return os.path.realpath(f"{os.path.dirname(os.path.abspath(__file__))}/..")


SCILPY_HOME = get_home()
SCILPY_ROOT = get_root()
