#!/usr/bin/env python3
from setuptools import setup

setup(
    name='mtil',
    version='0.0.1',
    packages=['mtil'],
    install_requires=[
        # the MAGICAL benchmark that this suite is meant to solve :)
        'magical @ git+https://github.com/qxcv/magical.git',
        'gym>=0.15.0',
        'Click>=7.0',
        'numpy>=1.17.4',
        # I have my own fork that I periodically make changes to
        ('rlpyt @ git+https://github.com/qxcv/rlpyt.git'
         '@d2965f7219a7a5d25c9de32102237164fea0e00c'),
        'torch>=1.4.0,<1.5.0',
        'torchvision>=0.4.2,<0.6.0',
        'pyprind>=2.11.2',
        # kornia fork, updated to allow me to pass in a border_mode for affine
        # augmentation
        ('kornia @ git+https://github.com/qxcv/kornia.git'
         '@c65d73c07c09ef56c415d1d059df63aae40031fb'),
        # for HP search
        'ray[tune,rllib]==0.8.4',
        'setproctitle>=1.1.10',
        'psutil>=5.6.5',
        'scikit-optimize>=0.7.4,<0.8.0',
        # for final experiments (& data collection)
        'PyYAML>=5.1.2',
        'jupyterlab==1.2.*',
        'pandas==0.25.*',
        'scikit-video==1.1.*',
    ])
