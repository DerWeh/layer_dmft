"""Fourier transforms for Matsubara Green's functions for iω_n ↔ τ."""
import logging

from collections import namedtuple

import numpy as np
import gftools as gt

LOGGER = logging.getLogger(__name__)
FourierFct = namedtuple('FourierFct', ['iw', 'tau'])


def bare_dft_iw2tau(gf_iw, beta):
    r"""Perform the discrete Fourier transform on the Hermitian function `gf_iw`.

    Hermitian means here that the function producing `gf_iw` has to fulfill
    :math:`f(-iω_n) = f^*(iω_n)`, thus the transform :math:`\tilde{f}(τ)` is
    real.
    It is assumed that the high frequency moments have been stripped from `gf_iw`,
    else oscillations will appear.

    Parameters
    ----------
    gf_iw : (..., N_iw) complex np.ndarray
        The function at **fermionic** Matsubara frequencies.
    beta : float
        The inverse temperature.

    Returns
    -------
    gf_tau : (..., 2*N_iw + 1) float np.ndarray
        The Fourier transform of `gf_iw` on the interval [0, β].

    """
    shape = gf_iw.shape
    N_tau = 2*shape[-1] + 1
    gf_iw_paded = np.zeros(shape[:-1] + (N_tau,), dtype=gf_iw.dtype)
    gf_iw_paded[..., 1:-1:2] = gf_iw
    gf_tau = np.fft.hfft(gf_iw_paded/beta)
    gf_tau = gf_tau[..., :N_tau]  # trim to [0, \beta]

    return gf_tau


def barde_dft_tau2iw(gf_tau, beta):
    r"""Perform the discrete Fourier transform on the real function `gf_tau`.

    `gf_tau` has to be given on the interval τ in [0, β].
    It is assumed that the high frequency moments have been stripped from `gf_iw`,
    else oscillations will appear.

    Parameters
    ----------
    gf_tau : (..., N_tau) float np.ndarray
        The function at imaginary frequencies.
    beta : float
        The inverse temperature.

    Returns
    -------
    gf_iw : (..., {N_iw - 1}/2) float np.ndarray
        The Fourier transform of `gf_tau` for positive Matsubara frequencies.

    """
    gf_tau_full_range = np.concatenate((-gf_tau[..., :-1], gf_tau), axis=-1)
    gf_iw = -beta*(np.fft.ihfft(gf_tau_full_range[..., :-1]))
    gf_iw = gf_iw[..., 1::2]

    return gf_iw


def dft_iw2tau(gf_iw, beta, moments=(1.,), dft_backend=bare_dft_iw2tau):
    """DFT from iω to τ, needing the 1/iw tail `moment`.

    Parameters
    ----------
    gf_iw : (..., N_iw) complex np.ndarray
        The function at **fermionic** Matsubara frequencies.
    beta : float
        The inverse temperature.
    moment : (...) float array_like or float
        High frequency moment of `gf_iw`.
    dft_backend : callable, optional
        The function called to perform the Fourier transform on the data stripped
        off its high frequency moment.

    Returns
    -------
    gf_tau : (..., 2*N_iw + 1) float np.ndarray
        The Fourier transform of `gf_iw` on the interval [0, β].

    """
    mom = get_gf_from_moments(moments, beta, N_iw=gf_iw.shape[-1])
    gf_iw = gf_iw - mom.iw
    gf_tau = dft_backend(gf_iw, beta)
    gf_tau += mom.tau
    return gf_tau


def dft_tau2iw(gf_tau, beta, moments=(1.,), dft_backend=barde_dft_tau2iw):
    """DFT from τ to iω, needing the 1/iw tail `moment`.

    Parameters
    ----------
    gf_tau : (..., N_tau) float np.ndarray
        The function at imaginary frequencies [0, β]
    beta : float
        The inverse temperature.
    moment : (...) float array_like or float
        High frequency moment of `gf_iw`.
    dft_backend : callable, optional
        The function called to perform the Fourier transform on the data stripped
        off its high frequency moment.

    Returns
    -------
    gf_iw : (..., {N_iw - 1}/2) float np.ndarray
        The Fourier transform of `gf_tau` for positive Matsubara frequencies.

    """
    m1 = -gf_tau[..., -1] - gf_tau[..., 0]
    if not np.allclose(m1, moments[0]):
        LOGGER.warning("Provided 1/z moment differs from jump: mom: %s jump: %s",
                       moments[0], m1)
    mom = get_gf_from_moments(moments, beta, N_iw=(gf_tau.shape[-1]-1)//2)
    gf_tau = gf_tau - mom.tau
    gf_iw = dft_backend(gf_tau, beta)
    gf_iw += mom.iw
    return gf_iw


def get_gf_from_moments(moments, beta, N_iw):
    """Green's function from `moments` on Matsubara axis and imaginary time.

    Parameters
    ----------
    moment : (n, ...) float array_like or float
        High frequency moment of `gf_iw`.
    beta : float
        The inverse temperature.
    N_iw : int
        Number of Matsubara frequencies

    Returns
    -------
    moments.iw : (..., N_iw) complex np.ndarray
        Matsubara Green's function calculated from the moments for positive
        frequencies.
    moments.tau : (..., 2*N_iw + 1) float np.ndarray
        Imaginary time Green's function calculated from the moments for τ in [0, β].

    """
    moments = np.asarray(moments)[..., np.newaxis]
    iws = gt.matsubara_frequencies(np.arange(N_iw), beta=beta)
    if len(moments) == 1:
        mom_iw = moments[0]/iws
        mom_tau = -.5*moments[0]
    elif len(moments) == 2:
        # FIXME: handle moments[0] = 0
        tau = np.linspace(0, beta, num=2*N_iw + 1, endpoint=True)
        pole = moments[1]/moments[0]
        mom_iw = moments[0]/(iws - pole)
        mom_tau = moments[0] * ft_pole2tau(tau, pole=pole, beta=beta)
    else:
        raise NotImplementedError()
    return FourierFct(iw=mom_iw, tau=mom_tau)


def ft_pole2tau(tau, pole, beta):
    """Fourier transform of 1/(iw - pole)."""
    assert np.all((tau >= 0) & (tau <= beta)), 'Only implemented for tau in [0, beta]'
    max_exp = np.where(tau > beta, beta - tau, beta)  # avoid overflows
    return -np.exp((beta - tau - max_exp)*pole)/(np.exp((beta - max_exp)*pole) + np.exp(-max_exp*pole))


