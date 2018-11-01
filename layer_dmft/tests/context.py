"""Provide top level module for tests."""
from __future__ import absolute_import
import os

from sys import path

PATH = os.path.abspath(os.path.dirname(__file__))
path.insert(0, os.path.join(PATH, os.pardir, os.pardir))

print(os.path.join(PATH, os.pardir, os.pardir))

print(path)
from layer_dmft import model, util, scatter
# import model
# import util
import gftools
# import scatter
