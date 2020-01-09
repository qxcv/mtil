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
        # going to keep this version of rlpyt relatively clean so that my
        # changes can be merged back upstream later on
        ('rlpyt @ git+https://github.com/qxcv/rlpyt.git'
         '#sha1=75e96cda433626868fd2a30058be67b99bbad810'),
        'torchvision>=0.4.2',
    ])
