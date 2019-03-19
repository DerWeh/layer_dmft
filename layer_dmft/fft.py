"""Fourier transforms for Matsubara Green's functions for iω_n ↔ τ."""
import numpy as np
import gftools as gt


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
    gf_tau_res : (..., 2*N_iw + 1) float np.ndarray
        The Fourier transform of `gf_iw` on the interval [0, β].

    """
    shape = gf_iw.shape
    N_tau = 2*shape[-1] + 1
    gf_iw_paded = np.zeros(shape[:-1] + (N_tau,), dtype=gf_iw.dtype)
    gf_iw_paded[..., 1:-1:2] = gf_iw
    gf_tau = np.fft.hfft(gf_iw_paded/beta)
    gf_tau_res = gf_tau[..., :N_tau]  # trim to [0, \beta]

    return gf_tau_res


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
    moments = np.asarray(moments)[..., np.newaxis]
    iws = gt.matsubara_frequencies(np.arange(gf_iw.shape[-1]), beta=beta)
    if len(moments) == 1:
        gf_iw = gf_iw - moments[0]/iws
        mom_tau = -.5*moments[0]
    elif len(moments) == 2:
        pole = moments[1]/moments[0]
        gf_iw = gf_iw - moments[0]/(iws - pole)
        tau = np.linspace(0, beta, num=2*gf_iw.shape[-1] + 1, endpoint=True)
        mom_tau = moments[0] * ft_pole2tau(tau, pole=pole, beta=beta)
    else:
        raise NotImplementedError()
    gf_tau = dft_backend(gf_iw, beta)
    gf_tau += mom_tau
    return gf_tau


def ft_pole2tau(tau, pole, beta):
    """Fourier transform of 1/(iw - pole)."""
    assert np.all((tau >= 0) & (tau <= beta)), 'Only implemented for tau in [0, beta]'
    max_exp = np.where(tau > beta, beta - tau, beta)  # avoid overflows
    return -np.exp((beta - tau - max_exp)*pole)/(np.exp((beta - max_exp)*pole) + np.exp(-max_exp*pole))


