# pylint: disable=redefined-outer-name
"""Test functionality concerning DMFT loop."""

from typing import NamedTuple
import pytest

import numpy as np
import gftools as gt

from .context import layer_dmft, model

Hubbard_Parameters = model.Hubbard_Parameters


class ExampleData(NamedTuple):
    """Data needed for calculations."""

    prm: Hubbard_Parameters
    iws: np.ndarray
    layer_data: layer_dmft.LayerIterData


@pytest.fixture(scope='module')
def l1_hartree() -> ExampleData:
    """Provide example data for non-magnetic Hartree Green's function."""
    prm = Hubbard_Parameters(N_l=1, lattice='bethe')
    prm.D = 1.
    prm.T = 0.02
    prm.U[:] = 2
    prm.h[:] = 0
    prm.mu[:] = -0.5
    prm.assert_valid()

    iws = gt.matsubara_frequencies(np.arange(2**10), beta=prm.beta)
    return ExampleData(prm, iws=iws, layer_data=layer_dmft.hartree_solution(prm, iw_n=iws))


@pytest.fixture(scope='module')
def l1_mag_hartree() -> ExampleData:
    """Provide example data for half-metallic Hartree Green's function."""
    print('create l1_mag_hartree')
    prm = Hubbard_Parameters(N_l=1, lattice='bethe')
    prm.D = 1.
    prm.T = 0.02
    prm.U[:] = 2
    prm.h[:] = .5
    prm.mu[:] = -1.5
    prm.assert_valid()

    iws = gt.matsubara_frequencies(np.arange(2**10), beta=prm.beta)
    return ExampleData(prm, iws=iws, layer_data=layer_dmft.hartree_solution(prm, iw_n=iws))


def test_mixed_siams(l1_mag_hartree: ExampleData):
    """Test if mixing gives correct hybridization function."""
    mixing = .3
    data = l1_mag_hartree

    siams_new = data.prm.get_impurity_models(data.iws, self_z=data.layer_data.self_iw,
                                             gf_z=data.layer_data.gf_iw, occ=data.layer_data.occ)
    # without self-energy
    gf0 = data.prm.gf0(data.iws)
    occ0 = data.prm.occ0(gf0, return_err=False)
    siams_old = data.prm.get_impurity_models(data.iws, self_z=0, gf_z=gf0, occ=occ0)

    siams_new, siams_old = list(siams_new), list(siams_old)
    siam_mixed = next(layer_dmft.mixed_siams(mixing, new=siams_new, old=siams_old))
    mixed_hybrid_tau = mixing*siams_new[0].hybrid_tau() + (1-mixing)*siams_old[0].hybrid_tau()
    assert np.allclose(siam_mixed.hybrid_tau(), mixed_hybrid_tau)
