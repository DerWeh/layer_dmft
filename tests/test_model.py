# coding: utf8
"""Tests for the Hubbards model related building blocks"""
from __future__ import absolute_import, unicode_literals

import pytest
import numpy as np

from .context import model
from .context import gftools


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


def test_SpinResolvedArray_iteration():
    """Assert that the array is iterable."""
    test = model.SpinResolvedArray(up=np.arange(9).reshape(3, 3),
                                   dn=np.arange(9).reshape(3, 3))
    for i, element in enumerate(test):
        pass


def test_compare_greensfunction():
    """Non-interacting and DMFT should give the same for self-energy=0."""
    prm = model.prm
    N = 13
    prm.T = 0.137
    prm.D = 1.  # half-bandwidth
    prm.mu = np.zeros(N)  # with respect to half filling
    prm.mu[N//2] = 0.45
    prm.V = np.zeros(N)
    prm.h = np.zeros(N)
    prm.h[N//2] = 0.9
    prm.U = np.zeros(N)
    t = 0.2
    prm.t_mat = np.zeros((N, N))
    diag, _ = np.diag_indices_from(prm.t_mat)
    sdiag = diag[:-1]
    prm.t_mat[sdiag+1, sdiag] = prm.t_mat[sdiag, sdiag+1] = t

    iw = gftools.matsubara_frequencies(np.arange(100), beta=prm.beta)
    gf0 = prm.gf0(iw)
    gf_dmft = prm.gf_dmft(iw, np.zeros_like(gf0))
    assert np.allclose(gf0, gf_dmft)
