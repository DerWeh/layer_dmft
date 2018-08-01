#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : tests/test_model.py
# Author            : Weh Andreas <andreas.weh@physik.uni-augsburg.de>
# Date              : 01.08.2018
# Last Modified Date: 01.08.2018
# Last Modified By  : Weh Andreas <andreas.weh@physik.uni-augsburg.de>
"""Tests for the Hubbards model related building blocks."""
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
    prm.hilbert_transform = model.hilbert_transform['bethe']
    t = 0.2
    prm.t_mat = np.zeros((N, N))
    diag, _ = np.diag_indices_from(prm.t_mat)
    sdiag = diag[:-1]
    prm.t_mat[sdiag+1, sdiag] = prm.t_mat[sdiag, sdiag+1] = t

    iw = gftools.matsubara_frequencies(np.arange(100), beta=prm.beta)
    gf0 = prm.gf0(iw)
    gf_dmft = prm.gf_dmft(iw, np.zeros_like(gf0))
    assert np.allclose(gf0, gf_dmft)


def test_2x2_matrix():
    """Compare with analytic inversion of (2, 2) matrix.
    
    Done for the 1D chain of sites.
    """
    prm = model.prm

    def gf_2x2(omega, t_mat, onsite_energys):
        assert np.all(np.diag(t_mat) == 0.), \
            "No diagonal elements for t_mat allowed"
        assert t_mat.shape == (2, 2)
        assert onsite_energys.shape == (2, )
        diag =  omega + onsite_energys
        norm = 1. / (np.prod(diag) - t_mat[0, 1]*t_mat[1, 0])
        gf = np.zeros_like(t_mat, dtype=np.complex)
        gf[0, 0] = diag[1]
        gf[1, 1] = diag[0]
        gf[0, 1] = -t_mat[0, 1]
        gf[1, 0] = -t_mat[1, 0]
        return norm*gf

    prm.T = 0.0137
    prm.t_mat = np.zeros((2, 2))
    prm.t_mat[0, 1] = prm.t_mat[1, 0] = 1.3
    prm.mu = np.array([0, 1.73])
    prm.h = np.array([0, -0.3])
    prm.U = 0
    prm.V = 0
    prm.hilbert_transform = model.hilbert_transform['chain']

    omegas = gftools.matsubara_frequencies(np.arange(100), prm.beta)
    gf_prm = prm.gf0(omegas, diagonal=False)
    gf_2x2_up = np.array([gf_2x2(iw, prm.t_mat, prm.onsite_energy(sigma=model.sigma.up))
                          for iw in omegas])
    gf_2x2_up = gf_2x2_up.transpose(1, 2, 0)  # adjuste axis order (2, 2, omegas)
    assert np.allclose(gf_2x2_up, gf_prm.up)
    gf_2x2_dn = np.array([gf_2x2(iw, prm.t_mat, prm.onsite_energy(sigma=model.sigma.dn))
                          for iw in omegas])
    gf_2x2_dn = gf_2x2_dn.transpose(1, 2, 0)  # adjuste axis order (2, 2, omegas)
    assert np.allclose(gf_2x2_dn, gf_prm.dn)
