"""Calculate conductivity according to Junya's notes and code.

Glossary
--------

curr
    Current
acorr, autocorrelation
    Correlation between the same quantity at different times.
n1
    single site

"""
import numpy as np
import gftools as gt
from gftools import pade

from layer_dmft import Hubbard_Parameters, model


def curr_acorr0_n1_iv0(prm: Hubbard_Parameters, *, occ=None, n_iw=2**16):
    """Calculate non-interacting bosonic Matsubara current-current correlation function.

    For the non-interacting case, only the zeroth Matsubara frequency
    :math:`iν_0 = 0` is finite, all other frequencies vanish. Thus only the
    zeroth frequency is calculated.

    Parameters
    ----------
    prm : Hubbard_Parameters
        The model.
    occ : (N_sp, N_l=1) float np.ndarray
        Used to include Hartree contribution if `prm.U ≠ 0`
    n_iw : int
        How many fermionic Matsubaras will be used for the summation.

    Returns
    -------
    curr_acorr_n1_iv : (N_sp, 1) complex np.ndarray
        The current-current correlation function.

    Notes
    -----
    For the non-interacting case it is in fact not a good idea, to use Matsubara
    summation. It would probably be more efficient to perform the Matsubara sum
    analytic and numerically integrate over the DOS.
    Furthermore, using Pade instead of Matsubara frequencies would be way more
    efficient.

    """
    # TODO: dynamically calculate the number of Matsubaras to use from temperature
    iws = gt.matsubara_frequencies(range(n_iw), beta=prm.beta)
    if occ is not None:
        occ = occ[::-1]
    xi = prm.onsite_energy(iws, hartree=occ)
    assert model.rev_dict_hilbert_transfrom[prm.hilbert_transform] == 'bethe'
    gf_d1 = gt.bethe_gf_d1_omega(np.add.outer(xi, iws), half_bandwidth=prm.D)
    delta_sum = 2*prm.T*np.sum(gf_d1 + 1./(iws + xi)**2, axis=-1).real
    return delta_sum - gt.fermi_fct_d1(xi, beta=prm.beta)


def curr_acorr_n1_iv(prm: Hubbard_Parameters, self_iw, occ, N_iv: int):
    """Calculate bosonic Matsubara current-current correlation function.

    Parameters
    ----------
    prm : Hubbard_Parameters
        The model.
    self_iw : (N_sp, N_l=1, N_iw) complex np.ndarray
        The local self-energy, for non-negative Matsubaras `iws.imag > 0`.
    occ : (N_sp, N_l=1) float np.ndarray
        Occupation number, needed to perform Matsubara sum corrections.
    N_iv : int
        Number of bosonic Matsubaras to calculate. Must be non-negative,
        only non-negative Matsubaras will be calculated.

    Returns
    -------
    curr_acorr_n1_iv : (N_sp, 1, N_iv) complex np.ndarray
        The current-current correlation function.

    """
    assert prm.N_l == 1
    assert occ.shape == self_iw.shape[:-1], "Shape mismatch of occupation and self-energy."
    N_iw = self_iw.shape[-1]
    # we need sum over positive and negative Matsubaras
    self_iw = np.concatenate((self_iw[..., ::-1].conj(), self_iw), axis=-1)
    iws = gt.matsubara_frequencies(range(-N_iw, N_iw), beta=prm.beta)
    assert model.rev_dict_hilbert_transfrom[prm.hilbert_transform] == 'bethe', \
        "Only Bethe lattice derivative implemented."
    gf_iw = prm.gf_dmft_s(iws, self_iw)
    xi_iw = (iws + prm.onsite_energy() - self_iw)
    # we assume as symmetric density of states, thus that ϵ^{(0)} = 0
    xi0 = prm.onsite_energy(hartree=occ[::-1])
    xi0_iw_inv = 1./(iws + xi0)
    gf_d1_iw = gt.bethe_gf_d1_omega(xi_iw, half_bandwidth=prm.D)

    def p_iv(n_b):
        # there is no `-0` so we have to treat that separately
        lmsk = Ellipsis, slice(None, -n_b if n_b != 0 else None)
        rmsk = Ellipsis, slice(n_b, None)
        d_xi_iw = xi_iw[lmsk] - xi_iw[rmsk]
        small_limit = abs(d_xi_iw) < 1e-8
        # only very few elements should be `small_limit` so we don't care about overhead
        res = gf_iw[lmsk] - gf_iw[rmsk]
        res[~small_limit] /= d_xi_iw[~small_limit]
        # symmetric, just in case
        res[small_limit] = .5*(gf_d1_iw[lmsk][small_limit] + gf_d1_iw[rmsk][small_limit])
        # Matsubara sum corrections
        delta_sum = prm.T*np.sum(res + xi0_iw_inv[lmsk]*xi0_iw_inv[rmsk], axis=-1)
        if n_b == 0:  # only relevant near half filling to have corrections
            return delta_sum - gt.fermi_fct_d1(xi0, beta=prm.beta)
        return delta_sum  # for N_l == 1 there are only corrections to iν_0

    return np.moveaxis(np.array([p_iv(n_b) for n_b in range(N_iv)]), 0, -1)


def conductivity_pade(curr_acorr_iv, beta, shift=0j, kind=None, **kwds):
    """Extract conductivity from current-current correlation via Pade.

    Analytic continuation is used to extrapolate the optical conductivity to
    frequency 0. Internally `pade.averaged` is used.

    Parameters
    ----------
    curr_acorr_iv : (..., N_iv) complex np.ndarray
        Current-current correlation function on non-negative bosonic
        Matsubara frequencies. Frequency 0 is required.
    beta : float
        Inverse temperature.
    shift : complex, optional
        Shift in the imaginary plain to help Pade and causality. Should not
        be necessary.
    kind : pade.KindGf, optional
        Parameter for Pade, which approximants to include in average.
    kwds
        Parameters passed to `pade.averaged`

    Returns
    -------
    conductivity.x, conductivity.err : complex
        The conductivity and its variance between different approximants

    """
    K_iv = curr_acorr_iv
    N_iv = K_iv.shape[-1]
    ivs = gt.matsubara_frequencies_b(range(1, N_iv), beta=beta)
    if kind is None:  # FIXME: find a good choice
        kind = pade.KindGf(0, N_iv-1)
    sigma = pade.averaged(np.array(0.+shift), ivs, fct_z=-(K_iv[1:] - K_iv[0])/ivs,
                          kind=kind, **kwds)
    return sigma
