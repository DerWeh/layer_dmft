"""Test conductivity."""
import numpy as np
from .context import conductivity as conduct, model


def test_U0_curr_acorr_n1():
    """Compare the results of `curr_acorr_n1_iv` with non-interacting case."""
    prm = model.Hubbard_Parameters(N_l=1, lattice='bethe')
    prm.D = 1.37
    prm.mu[:] = .136
    prm.T = 0.01
    prm.assert_valid()
    assert np.all(prm.U == 0)
    K_iv = conduct.curr_acorr_n1_iv(
        prm, self_iw=np.zeros((1, 1, 1024), dtype=complex),
        occ=np.zeros((1, 1)),  # U=0 thus occ has no effect
        N_iv=10
    )
    assert np.allclose(K_iv.imag, 0), "Correlation is real quantity."
    # accuracy depends on temperature!
    # Finite values are due to truncation of Matsubara sums
    assert np.allclose(K_iv[..., 1:], 0, atol=1e-4), "For U=0, only zeroth frequency contributes."
