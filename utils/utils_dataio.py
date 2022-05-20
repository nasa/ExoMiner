""" General utility functions. """

# 3rd party
import json
import yaml


def is_jsonable(x):
    """ Test if object is JSON serializable.

    :param x: object
    :return:
    """

    try:
        json.dumps(x)
        return True
    except:
        return False


def is_yamlble(x):
    """ Test if object is JSON serializable.

    :param x: object
    :return:
    """

    try:
        yaml.dump(x)
        return True
    except:
        return False
