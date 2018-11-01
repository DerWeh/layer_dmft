#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : tests/test_scatter.py
# Author            : Weh Andreas <andreas.weh@physik.uni-augsburg.de>
# Date              : 20.09.2018
# Last Modified Date: 20.09.2018
# Last Modified By  : Weh Andreas <andreas.weh@physik.uni-augsburg.de>
"""Tests for scattering formulation of inhomogeneous problems."""
from __future__ import absolute_import, unicode_literals

import pytest
import numpy as np
import hypothesis.strategies as st

from hypothesis import given
from hypothesis.extra.numpy import arrays as st_arrays

from .context import scatter


def test_T_matrix():
    """Compera formulations of the T-matrix."""
    pot1 = np.diagflat([0, ]*5 + [1.49, ]*5)
    pot2 = np.diagflat([0, ]*5 + [1, ] + [0, ]*4)
    gf1 = np.eye(10, dtype=complex) + 1e-4j
    gf1 += np.diagflat([1.37, ]*9, k=1)
    gf1 += gf1.T
    gf2 = 0.3*np.eye(10, dtype=complex) + 1e-4j
    gf2 += 3.89
    for gf, pot in zip((gf1, gf2), (pot1, pot2)):
        t_mat = scatter.t_matrix(g_hom=gf, potential=pot)
        t_mat_comp1 = np.linalg.inv(np.eye(*gf.shape) - pot@gf) @ pot
        t_mat_comp2 = pot @ np.linalg.inv(np.eye(*pot.shape) - gf@pot)
        assert np.allclose(t_mat, t_mat_comp1)
        assert np.allclose(t_mat, t_mat_comp2)
