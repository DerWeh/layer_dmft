#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : setup.py
# Author            : Weh Andreas <andreas.weh@physik.uni-augsburg.de>
# Date              : 03.05.2019
# Last Modified Date: 03.05.2019
# Last Modified By  : Weh Andreas <andreas.weh@physik.uni-augsburg.de>
"""Setup to use `layer_dmft` as module."""
from setuptools import setup

import versioneer


def readme():
    with open('README.rst') as file_:
        return file_.read()


setup(
    name="layer_dmft",
    version=versioneer.get_version(),
    description="r-DMFT code for layered heterostructures",
    long_description=readme(),
    keywords=r"DMFT",
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.0',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    # url="https://github.com/DerWeh/gftools",
    # project_urls={
    #     "Documentation": "https://derweh.github.io/gftools/",
    #     "ReadTheDocs": "https://gftools.readthedocs.io/en/latest/",
    #     "Source Code": "https://github.com/DerWeh/gftools",
    # },
    author="Weh",
    author_email="andreas.weh@physik.uni-augsburg.de",
    cmdclass=versioneer.get_cmdclass(),
    packages=['layer_dmft'],
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'xarray',
        'h5netcdf',  # currently needed to save complex arrays
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'hypothesis'],
)
