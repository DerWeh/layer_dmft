"""Fourier transforms for Matsubara Green's functions for iω_n ↔ τ."""
import logging
from functools import partial, lru_cache

from collections import namedtuple

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import gftools as gt

LOGGER = logging.getLogger(__name__)
FourierFct = namedtuple('FourierFct', ['iw', 'tau'])


def _has_nan(x):
    """Check for `np.nan` in `x`."""
    x = x.reshape(-1)
    return np.isnan(np.dot(x, x))


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


def soft_dft_iw2tau(gf_iw, beta):
    r"""Perform the discrete Fourier transform on the Hermitian function `gf_iw`.

    Add a tail letting `gf_iw` go to 0. The tail is just a cosine function to
    exactly hit the 0.
    This is unphysical but suppresses oscillations. This methods should be used
    with care, as it might hide errors.

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
    gf_iw_extended = np.zeros(shape[:-1] + (shape[-1]*2,), dtype=gf_iw.dtype)
    gf_iw_extended[..., :shape[-1]] = gf_iw
    tail_range = np.linspace(0, np.pi, num=shape[-1])
    tail = .5*(np.cos(tail_range) + 1.)
    LOGGER.debug("Remaining tail approximated by 'cos': %s", gf_iw[..., -1:])
    gf_iw_extended[..., shape[-1]:] = tail*gf_iw[..., -1:]
    N_tau = 2*gf_iw_extended.shape[-1] + 1
    gf_iw_paded = np.zeros(shape[:-1] + (N_tau,), dtype=gf_iw.dtype)
    gf_iw_paded[..., 1:-1:2] = gf_iw_extended
    gf_tau = np.fft.hfft(gf_iw_paded/beta)
    gf_tau = gf_tau[..., :N_tau:2]  # trim to [0, \beta]

    return gf_tau


def bare_dft_tau2iw(gf_tau, beta):
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


@partial(np.vectorize, signature='(n),(n),(n),(m)->(m)', excluded={'s'})
def interpolate(x_in, fct_in, fct_err, x_out, s=None):
    """Calculate complex interpolation of `fct_in` and evaluate it at `x_out`."""
    w = 1./fct_err
    fct_out = UnivariateSpline(x_in, fct_in, w=w, s=s)(x_out)
    return fct_out


def interpolated_dft_tau2iw(gf_tau, beta, gf_tau_err):
    """Perform discrete Fourier transform on the real function `gf_tau`.

    `gf_tau` has to be given on the interval τ in [0, β].
    It is assumed that the high frequency moments have been stripped from `gf_iw`,
    else oscillations will appear.
    An smoothening spline interpolation of `gf_tau` is performed to better
    approximate the continuous Fourier integral.

    Parameters
    ----------
    gf_tau : (..., N_tau) float np.ndarray
        The function at imaginary frequencies.
    beta : float
        The inverse temperature.
    gf_tau_err : (..., N_tau) float np.ndarray
        The error of `gf_tau` used to improve the smoothened interpolation.

    Returns
    -------
    gf_iw : (..., {N_iw - 1}/2) float np.ndarray
        The Fourier transform of `gf_tau` for positive Matsubara frequencies.

    """
    SMOTHENING = 0.01  # small value is important for noisy data
    MULT_TAU_POINTS = 1024
    tau_points = gf_tau.shape[-1]
    # FIXME: pad data to ensure smoothness at boundary
    tau = np.linspace(0, 1., num=tau_points, endpoint=True)
    tau_interp = np.linspace(0, 1., num=MULT_TAU_POINTS*(tau_points - 1) + 1, endpoint=True)
    gf_tau_interp = interpolate(x_in=tau, fct_in=gf_tau, fct_err=gf_tau_err,
                                x_out=tau_interp, s=SMOTHENING*tau_points)
    gf_iw_long = bare_dft_tau2iw(gf_tau_interp, beta=beta)
    return gf_iw_long[..., :(tau_points - 1)//2]


def dft_iw2tau(gf_iw, beta, moments=(1.,), dft_backend=soft_dft_iw2tau):
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
    order = len(moments)  # these orders are done by moments
    # fit of tail for real part (even order) and imaginary part (odd order)
    tail1 = fit_iw_tail(gf_iw, beta=beta, order=order+1)
    tail2 = fit_iw_tail(gf_iw, beta=beta, order=order+2)
    ensure_fit_causality(mom, tail1, tail2)
    gf_iw = gf_iw - tail1.iw - tail2.iw
    gf_tau = dft_backend(gf_iw, beta)
    gf_tau += mom.tau + tail1.tau + tail2.tau
    return gf_tau


def ensure_fit_causality(mom: FourierFct, tail1: FourierFct, tail2: FourierFct):
    """Set tails which violate causality to 0.

    Parameters
    ----------
    mom : FourierFct
        Green's function obtained from specified moments.
    tail1, tail2 : FourierFct
        Fitted tail of the form `c/(iw)^{n+i}`, where `n = len(mom)` and `i = 1 (2)`
        for `tail1` (`tail2`).

    """
    # after the numeric data ends, only tail exists, its imaginary part should
    # at no point be positive
    if np.all((mom.iw[..., -1] + tail1.iw[..., -1] + tail2.iw[..., -1]).imag <= 0):
        return  # calculated tail is causal, nothing to do
    # the current implementation is dependent on the implementation of the
    # tails it is assumed that on is purely real and one purely imaginary
    # (`c/(iw)^n` with `c` real) and that `c<0` if `tail<0`
    imag_tail = tail2 if np.any(np.iscomplex(tail2.iw)) else tail1
    positive = imag_tail.iw[..., -1] > 0
    LOGGER.warning("The fitted moments corresponding to indices %s result in non-causal"
                   "tails and are therefore ignored!", positive)
    imag_tail.iw[positive], imag_tail.tau[positive] = 0, 0  # set according fitted tails to 0
    if np.any(mom.iws[..., -1] < 0):  # this should not happen!
        LOGGER.warning("Tail calculated from moments is non-causal, "
                       "apparently some incorrect moments were provided.")


def dft_tau2iw(gf_tau, beta, moments=(1.,), dft_backend=bare_dft_tau2iw):
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


def get_gf_from_moments(moments, beta, N_iw) -> FourierFct:
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
        return FourierFct(iw=moments[0]/iws, tau=-.5*moments[0])
    if len(moments) == 2:
        if np.any(moments[0] == 0.):
            if np.all(moments[1] == 0.) and np.all(moments[0] == 0.):
                return FourierFct(iw=0, tau=0)
            # TODO: TEST THIS!
            tau = np.linspace(0, beta, num=2*N_iw + 1, endpoint=True)
            mom_iw = np.zeros((moments[0].size, N_iw), dtype=iws.dtype)
            mom_tau = np.zeros((moments[0].size, tau.size), dtype=tau.dtype)
            # cases where mom[0] == 0:
            mom_is0 = moments[0, :, 0] == 0
            mom_iw[mom_is0] = moments[1, mom_is0]/iws**2
            mom_tau[mom_is0] = moments[1, mom_is0]*(.5*tau + .25*beta)
            # cases where mom[0] != 0:
            pole = moments[1, ~mom_is0]/moments[0, ~mom_is0]
            mom_iw[~mom_is0] = moments[0, ~mom_is0]/(iws - pole)
            mom_tau[~mom_is0] = moments[0, ~mom_is0] * ft_pole2tau(tau, pole=pole, beta=beta)
            return FourierFct(iw=mom_iw, tau=mom_tau)
        tau = np.linspace(0, beta, num=2*N_iw + 1, endpoint=True)
        pole = moments[1]/moments[0]
        mom_iw = moments[0]/(iws - pole)
        mom_tau = moments[0] * ft_pole2tau(tau, pole=pole, beta=beta)
        return FourierFct(iw=mom_iw, tau=mom_tau)
    raise NotImplementedError()


def fit_iw_tail(gf_iw, beta, order) -> FourierFct:
    """Fit the tail of `gf_iw` with function behaving as (iw)^{-`order`}.

    Parameters
    ----------
    gf_iw : (..., N_iw) complex np.ndarray
        The function at **fermionic** Matsubara frequencies.
    beta : float
        The inverse temperature `beta` = 1/T.
    order : int
        Leading order of the high-frequency behavior of the tail.

    Returns
    -------
    fit_iw_tail.iw : (..., N_iw) complex np.ndarray
        The tail fit for the same frequencies as `gf_iw`.
    fit_iw_tail.tau : (..., 2*N_iw + 1) float np.ndarray
        The Fourier transform of the tail for τ ∈ [0, β].

    Raises
    ------
    RuntimeError
        If the Fourier transform of the tail contains `np.nan`. This should be
        fixed and never occur. It indicates a overflow in `get_gf_from_moments()`.

    """
    N_iw = gf_iw.shape[-1]
    odd = order % 2
    # CC = 2.*order  # shift to make the function small for low frequencies
    CC = 0.
    iws = gt.matsubara_frequencies(np.arange(N_iw), beta=beta)
    tau = np.linspace(0, beta, num=2*N_iw + 1, endpoint=True)

    def to_float(number):
        # scipy only handles np.float64
        return (number.imag if odd else number.real).astype(np.float64)

    norm_tail_iw = .5 * ((iws + CC)**-order + (iws - CC)**-order)  # tail with amplitude 1
    sigma = to_float(iws**(-order-2))  # next order correction that is odd/even

    moment, err = _fitting(iws.imag, fit_iw=to_float(norm_tail_iw),
                           gf_iw=to_float(gf_iw), sigma=sigma)
    LOGGER.info('Amplitude of fit (order %s): %s ± %s', order, moment, err)
    moment = moment[..., np.newaxis]
    gf_tau_fct = get_order_n_pole(order)
    tail_tau = moment*.5*(gf_tau_fct(tau, CC, beta) + gf_tau_fct(tau, -CC, beta))
    if _has_nan(tail_tau):
        raise RuntimeError("Calculation of tail-fit failed. Most likely a overflow occurred.")
    return FourierFct(iw=moment*norm_tail_iw, tau=tail_tau)


@partial(np.vectorize, otypes=[float, float], signature='(n),(n),(n),(n)->(),()')
def _fitting(iws, fit_iw, gf_iw, sigma) -> (float, float):
    """Perform fits, all input data need to be `np.float64`."""
    if np.all(gf_iw == 0):
        return 0, 0
    START = 0  # from where to fit the tail
    moment, pcov = curve_fit(
        lambda xx, moment: moment*fit_iw[START:],
        # relies that xx is always the same
        iws[START:], ydata=gf_iw[START:],
        p0=(1.,), sigma=sigma[START:]
    )
    moment, err = moment.squeeze(), np.sqrt(pcov.squeeze())
    if moment > 5000:  # cutoff, this seems unreasonable large
        LOGGER.warning("Amplitude of fit exceeds cutoff and seems unreasonable large (%s ± %s)!"
                       "\nFit is ignored.")
        return 0, np.nan
    if (moment > 500 and pcov/err > 1e-2) or pcov/err > 0.1:
        # For very large fitted moments, we require increased accuracy
        LOGGER.warning("Error of moment large (%s ± %s)! Fit is ignored.",
                       moment, err)
        return 0, np.nan
    return moment, err


def ft_pole2tau(tau, pole, beta):
    """Fourier transform of 1/(iw - pole)."""
    assert np.all((tau >= 0) & (tau <= beta)), 'Only implemented for tau in [0, beta]'
    max_exp = np.where(tau > beta, beta - tau, beta)  # avoid overflows
    return -np.exp((beta - tau - max_exp)*pole)/(np.exp((beta - max_exp)*pole) + np.exp(-max_exp*pole))


@lru_cache(maxsize=6)
def get_order_n_pole(order):
    """Generate function to calculate the Fourier transform `order`-order pole.

    The Fourier transform of :math:`{(z-ϵ)}^{n}` is calculate, where `ϵ` is the
    position of the pole and `n` the order of the pole.

    Parameters
    ----------
    order : int
        The order of the pole

    Returns
    -------
    order_n_pole : Callable
        The function (tau, pole, beta)->gf_tau calculating the Fourier transform.

    """
    import theano
    import theano.tensor as T
    from theano.ifelse import ifelse
    from math import factorial

    pole = T.dscalar('pole')
    beta = T.dscalar('beta')
    # tau = T.dscalar('tau')
    tau = T.dscalar('tau')
    fermi_fct = (1 + T.tanh(-beta*pole/2))/2

    gf_tau = ifelse(
        pole > 0,  # avoid overflows asserting negative exponent
        -(1 - fermi_fct)*T.exp(-pole*tau),
        -fermi_fct*T.exp(pole*(beta-tau)),
    )
    n_gf_tau = gf_tau
    for __ in range(order-1):
        # n_gf_tau = T.grad(n_gf_tau, pole)
        n_gf_tau = T.jacobian(n_gf_tau, pole)
    n_gf_tau = n_gf_tau / factorial(order-1)
    # resuts, __ = theano.scan(n_gf_tau.)
    func = theano.function([tau, pole, beta], n_gf_tau)
    return np.vectorize(func, otypes=[np.float])
