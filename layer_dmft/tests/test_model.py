#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : tests/test_model.py
# Author            : Weh Andreas <andreas.weh@physik.uni-augsburg.de>
# Date              : 01.08.2018
# Last Modified Date: 01.08.2018
# Last Modified By  : Weh Andreas <andreas.weh@physik.uni-augsburg.de>
"""Tests for the Hubbard model related building blocks."""
from __future__ import absolute_import, unicode_literals

import pytest
import numpy as np

import gftools

from .context import model, util

SpinResolvedArray = util.SpinResolvedArray


def test_SpinResolvedArray_creation():
    """Basic test that `SpinResolvedArray` constructor creates suitable array."""
    assert np.all(SpinResolvedArray([1, 2]) == np.array([1, 2]))
    assert np.all(SpinResolvedArray(up=1, dn=2) == np.array([1, 2]))


def test_SpinResolvedArray_access():
    """Basic test for accessing elements of `SpinResolvedArray`s."""
    updata = np.arange(0, 7)
    dndata = np.arange(0, 7)
    test_array = SpinResolvedArray(up=updata, dn=dndata)
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
    assert not isinstance(SpinResolvedArray([1, 2]).up, SpinResolvedArray)
    spin_array = SpinResolvedArray(up=np.arange(9), dn=np.arange(9))
    assert not isinstance(spin_array.up, SpinResolvedArray)
    # assert type(spin_array[slice(1, None, 1)])\
    #     is not SpinResolvedArray


def test_SpinResolvedArray_iteration():
    """Assert that the array is iterable."""
    test = SpinResolvedArray(up=np.arange(9).reshape(3, 3),
                             dn=np.arange(9).reshape(3, 3))
    for __ in test:
        pass


def test_compare_greensfunction():
    """Non-interacting and DMFT should give the same for self-energy=0."""
    N = 13
    prm = model.Hubbard_Parameters(N)
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
    prm.t_mat = model.hopping_matrix(N, nearest_neighbor=t)

    iw = gftools.matsubara_frequencies(np.arange(100), beta=prm.beta)
    gf0 = prm.gf0(iw)
    gf_dmft = prm.gf_dmft_s(iw, np.zeros_like(gf0))
    assert np.allclose(gf0, gf_dmft)


def test_non_interacting_siam():
    """Compare that the SIAM yields the correct local Green's function."""
    N_l = 7
    prm = model.Hubbard_Parameters(N_l)
    prm.T = 0.137
    prm.D = 1.3  # half-bandwidth
    prm.mu = np.linspace(-.78, .6, num=N_l)
    prm.h = np.linspace(-1.47, .47, num=N_l)
    prm.hilbert_transform = model.hilbert_transform['bethe']
    t = 0.2
    prm.t_mat = model.hopping_matrix(N_l, nearest_neighbor=t)

    iw = gftools.matsubara_frequencies(np.arange(1024), beta=prm.beta)
    gf_layer = prm.gf0(iw)
    occ = prm.occ0(gf_layer)
    siams = prm.get_impurity_models(iw, self_z=0, gf_z=gf_layer, occ=occ.x)

    for lay, siam in enumerate(siams):
        assert np.allclose(gf_layer[:, lay], siam.gf0())
        # check that gf0 and gf_s coincide
        assert np.allclose(siam.gf0(), siam.gf_s(0))


def test_2x2_matrix():
    """Compare with analytic inversion of (2, 2) matrix.

    Done for the 1D chain of sites.
    """
    prm = model.Hubbard_Parameters(2)

    def gf_2x2(omega, t_mat, onsite_energys):
        assert np.all(np.diag(t_mat) == 0.), \
            "No diagonal elements for t_mat allowed"
        assert t_mat.shape == (2, 2)
        assert onsite_energys.shape == (2, )
        diag = omega + onsite_energys
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
    prm.D = None
    prm.hilbert_transform = model.hilbert_transform['chain']

    omegas = gftools.matsubara_frequencies(np.arange(100), prm.beta)
    gf_prm = prm.gf0(omegas, diagonal=False)
    gf_2x2_up = np.array([gf_2x2(iw, prm.t_mat, prm.onsite_energy(sigma=model.SIGMA.up))
                          for iw in omegas])
    gf_2x2_up = gf_2x2_up.transpose(1, 2, 0)  # adjuste axis order (2, 2, omegas)
    assert np.allclose(gf_2x2_up, gf_prm.up)
    gf_2x2_dn = np.array([gf_2x2(iw, prm.t_mat, prm.onsite_energy(sigma=model.SIGMA.dn))
                          for iw in omegas])
    gf_2x2_dn = gf_2x2_dn.transpose(1, 2, 0)  # adjuste axis order (2, 2, omegas)
    assert np.allclose(gf_2x2_dn, gf_prm.dn)
