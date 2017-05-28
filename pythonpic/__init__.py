# coding=utf-8
"""A work in progress particle-in-cell code written in Python, optimized for speed as well as readability."""

from . import classes
from . import algorithms
from . import visualization
from . import configs
import sys
import warnings

if sys.version_info[:2] < (3, 6):
    warnings.warn("PythonPIC does not support Python 3.5 and below")