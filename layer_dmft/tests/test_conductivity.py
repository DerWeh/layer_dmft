"""Test conductivity."""
import numpy as np
import gftools as gt

from .context import conductivity as conduct, model


def test_curr_acorr0_n1_low_temp():
    """Compare numeric Matsubara sum with Sommerfeld expansion for low T."""
    prm = model.Hubbard_Parameters(N_l=1, lattice='bethe')
    prm.D = 1.37
    prm.mu[:] = .536
    prm.T = 0.001
    prm.assert_valid()
    K_iv0 = conduct.curr_acorr0_n1_iv0(prm, occ=np.zeros((1, 1)))
    xi = prm.onsite_energy()
    aT0 = gt.bethe_dos(xi, half_bandwidth=prm.D)
    at2 = -1./np.pi*gt.bethe_gf_d2_omega(xi+1e-16j, half_bandwidth=prm.D).imag
    sommerfeld = aT0 + (np.pi*prm.T)**2/6.*at2
    assert np.allclose(K_iv0, sommerfeld)

    prm = model.Hubbard_Parameters(N_l=1, lattice='bethe')
    prm.D = 1.37
    prm.mu[:] = 0
    prm.T = 0.001
    prm.assert_valid()
    K_iv0 = conduct.curr_acorr0_n1_iv0(prm, occ=np.zeros((1, 1)))
    xi = prm.onsite_energy()
    aT0 = gt.bethe_dos(xi, half_bandwidth=prm.D)
    at2 = -1./np.pi*gt.bethe_gf_d2_omega(xi+1e-16j, half_bandwidth=prm.D).imag
    sommerfeld = aT0 + (np.pi*prm.T)**2/6.*at2
    assert np.allclose(K_iv0, sommerfeld)


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
    assert np.allclose(K_iv[..., 0], conduct.curr_acorr0_n1_iv0(prm, occ=np.zeros([1, 1])))
