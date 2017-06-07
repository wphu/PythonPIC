# coding=utf-8
from setuptools import setup

NAME = "pythonpic"

VERSION = "0.3"

setup(
    name='pythonpic',
    version=VERSION,
    packages=[NAME],
    url='https://github.com/StanczakDominik/PythonPIC',
    license='',
    author='Dominik Stańczak',
    author_email='stanczakdominik@gmail.com',
    description='A particle-in-cell code written in Python optimized for speed as well as readability.',
    python_requires='>=3.6', install_requires=['matplotlib', 'numpy', 'scipy', 'h5py', 'pytest', 'pandas', 'numba']
    )
