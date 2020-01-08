#!/usr/bin/env python3
from setuptools import setup

setup(
    name='mtil',
    version='0.0.1',
    packages=['mtil'],
    install_requires=[
        'gym~=0.15.0',
        'Click~=7.0',
        'numpy~=1.17.4',
        'tensorflow>=2.0.0',
    ])
