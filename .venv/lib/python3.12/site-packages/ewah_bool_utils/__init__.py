"""Top-level package for EWAH Bool Utils."""


from ewah_bool_utils.ewah_bool_wrap import *


def get_include():
    """
    Returns the directory that contains ewah*.h headers
    """
    import os

    return os.path.abspath(os.path.join(os.path.dirname(__file__), "cpp"))
