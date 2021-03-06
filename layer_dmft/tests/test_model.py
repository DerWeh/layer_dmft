#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : test_model.py
# Author            : Weh Andreas <andreas.weh@physik.uni-augsburg.de>
# Date              : 01.08.2018
# Last Modified Date: 09.05.2019
# Last Modified By  : Weh Andreas <andreas.weh@physik.uni-augsburg.de>
"""Tests for the Hubbard model related building blocks."""
from __future__ import absolute_import, unicode_literals

import pytest
import numpy as np
import xarray as xr

import gftools as gt

from .context import model, util

Dim = util.Dimensions
SIGMA = model.SIGMA


def test_compare_greensfunction():
    """Non-interacting and DMFT should give the same for self-energy=0."""
    N = 13
    prm = model.Hubbard_Parameters(N, lattice='bethe')
    prm.T = 0.137
    prm.D = 1.  # half-bandwidth
    prm.mu[N//2] = 0.45
    prm.h[N//2] = 0.9
    t = 0.2
    prm.t_mat = model.hopping_matrix(N, nearest_neighbor=t)

    iw = gt.matsubara_frequencies(np.arange(100), beta=prm.beta)
    gf0 = prm.gf0(iw)
    gf_dmft = prm.gf_dmft_s(iw, np.zeros_like(gf0))
    assert np.allclose(gf0, gf_dmft)


def test_non_interacting_siam():
    """Compare that the SIAM yields the correct local Green's function."""
    N_l = 7
    prm = model.Hubbard_Parameters(N_l, lattice='bethe')
    prm.T = 0.137
    prm.D = 1.3  # half-bandwidth
    prm.mu = np.linspace(-.78, .6, num=N_l)
    prm.h = np.linspace(-1.47, .47, num=N_l)
    t = 0.2
    prm.t_mat = model.hopping_matrix(N_l, nearest_neighbor=t)

    iw = model.matsubara_frequencies(range(1024), beta=prm.beta)
    gf_layer = prm.gf0(iw)
    occ = prm.occ0(gf_layer)
    siams = prm.get_impurity_models(iw, self_z=xr.zeros_like(gf_layer), gf_z=gf_layer, occ=occ.x)

    for lay, siam in enumerate(siams):
        assert np.allclose(gf_layer[:, lay], siam.gf0())
        # check that gf0 and gf_s coincide
        assert np.allclose(siam.gf0(), siam.gf_s(0))


def test_2x2_matrix():
    """Compare with analytic inversion of (2, 2) matrix.

    Done for the 1D chain of sites.
    """
    prm = model.Hubbard_Parameters(2, lattice='chain')

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
    prm.D = None
    prm.assert_valid()

    omegas = gt.matsubara_frequencies(np.arange(100), prm.beta)
    gf_prm = prm.gf0(omegas, diagonal=False)
    e_onsite = prm.onsite_energy()
    gf_2x2_up = np.array([gf_2x2(iw, prm.t_mat, e_onsite.sel({Dim.sp: 'up'}).values)
                           for iw in omegas])
    gf_2x2_up = gf_2x2_up.transpose(1, 2, 0)  # adjuste axis order (2, 2, omegas)
    assert np.allclose(gf_2x2_up, gf_prm.sel({Dim.sp: 'up'}))
    gf_2x2_dn = np.array([gf_2x2(iw, prm.t_mat, e_onsite.sel({Dim.sp: 'dn'}).values)
                          for iw in omegas])
    gf_2x2_dn = gf_2x2_dn.transpose(1, 2, 0)  # adjuste axis order (2, 2, omegas)
    assert np.allclose(gf_2x2_dn, gf_prm.sel({Dim.sp: 'dn'}))


def test_particle_hole_symmtery():
    """Compare particle hole-symmetric case in the most simple example N_l=1."""
    prm = model.Hubbard_Parameters(1, lattice='bethe')
    prm.T = 0.0137
    prm.U[:] = 6
    prm.D = 1.37
    prm.assert_valid()

    iws = model.matsubara_frequencies(range(1024), beta=prm.beta)
    assert prm.onsite_energy(hartree=[.5,]) == 0.
    occ = prm.occ0(prm.gf0(iws, hartree=[.5,]), hartree=[.5,])
    assert occ.x - occ.err <= .5 <= occ.x + occ.err

    omega = np.linspace(-2*prm.D, 2*prm.D, num=1000) + 1e-6j
    gf_iw = prm.gf0(omega, hartree=[.5])
    assert np.allclose(gf_iw, gt.bethe_gf_omega(omega, half_bandwidth=prm.D))
