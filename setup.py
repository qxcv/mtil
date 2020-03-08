#!/usr/bin/env python3
from setuptools import setup

setup(
    name='mtil',
    version='0.0.1',
    packages=['mtil'],
    install_requires=[
        'gym>=0.15.0',
        'Click>=7.0',
        'numpy>=1.17.4',
        # I have my own fork that I periodically make changes to
        ('rlpyt @ git+https://github.com/qxcv/rlpyt.git'
         '#sha1=d2965f7219a7a5d25c9de32102237164fea0e00c'),
        'torch>=1.4.0',
        'torchvision>=0.4.2',
        # kornia fork, updated to allow me to pass in a border_mode for affine
        # augmentation
        ('kornia @ git+https://github.com/qxcv/kornia.git'
         '#sha1=947fa4e34bb322be462c48848143e4d8d9063547'),
    ])
