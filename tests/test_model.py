# coding: utf8
"""Tests for the Hubbards model related building blocks"""
from __future__ import absolute_import, unicode_literals

import pytest
import numpy as np

from .context import model


def test_SpinResolvedArray_creation():
    assert np.all(model.SpinResolvedArray([1, 2]) == np.array([1, 2]))
    assert np.all(model.SpinResolvedArray(up=1, dn=2) == np.array([1, 2]))


def test_SpinResolvedArray_access():
    updata = np.arange(0, 7)
    dndata = np.arange(0, 7)
    test_array = model.SpinResolvedArray(up=updata,
                                         dn=dndata)
    assert np.all(test_array.up == updata)
    assert np.all(test_array[0] == updata)
    assert np.all(test_array['up'] == updata)
    assert np.all(test_array['up', ...] == updata)
    assert np.all(test_array['up', 2:5:-2] == updata[2:5:-2])

    assert np.all(test_array.dn == dndata)
    assert np.all(test_array[1] == dndata)
    assert np.all(test_array['dn'] == dndata)
    assert np.all(test_array['dn', ...] == dndata)
    assert np.all(test_array['dn', 2:5:-2] == dndata[2:5:-2])


def test_SpinResolvedArray_elements():
    """Assert that the elements of SpinResolvedArray are regular arrays."""
    assert type(model.SpinResolvedArray([1, 2]).up) is not model.SpinResolvedArray
    assert type(model.SpinResolvedArray(up=np.arange(9), dn=np.arange(9)).up) \
        is not model.SpinResolvedArray
