# encoding: utf-8
"""Handles the charge self-consistency loop of the combined scheme."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from builtins import (ascii, bytes, chr, dict, filter, hex, input, int, map,
                      next, oct, open, pow, range, round, str, super, zip)
from functools import partial

import numpy as np

from scipy import optimize

import capacitor_formula

import gftools as gf
import gftools.matrix as gfmatrix

from model import prm, sigma, SpinResolvedArray, spins

# V_DATA = 'loop/Vsteps.dat'
# DMFT_PARAM = 'layer_hb_param.init'
# OUTPUT = 'output.txt'


# FIXME
layers = np.arange(40)
# define get_V
e_schot = np.ones_like(layers) * 0.4
get_V = partial(capacitor_formula.potential_energy_vector,
                e_schot=e_schot, layer_labels=layers)


def invert_gf_0(omega, gf_0_inv, half_bandwidth):
    """Return the diagonal elements of the Green's function in real space.

    Parameters
    ----------
    omega : array(complex)
        Frequencies at which the Green's function is evaluated.
    gf_0_inv : array(complex, complex)
        Square matrix containing the inverse elements of the Green's function
        :math:`G(ω, ϵ)` with ω and ϵ split off.
    half_bandwidth : float
        Half bandwidth of the semi circular Bethe DOS.

    Returns
    -------
    gf_diag : array(complex)
        The diagonal elements of the real space Green's function.

    """
    gf_0_inv = gf_0_inv.copy()
    diag = np.diag_indices_from(gf_0_inv)
    gf_0_inv_diag = np.diag(gf_0_inv).copy()
    omega = np.asarray(omega, dtype=np.complex256)
    gf_diag = np.zeros((len(gf_0_inv), len(omega)), dtype=np.complex256)
    for i, wi in enumerate(omega):
        gf_0_inv[diag] = gf_0_inv_diag + wi
        rv_inv, h, rv = gfmatrix.decompose_gf_omega(gf_0_inv)
        h_bar = gf.bethe_hilbert_transfrom(h, half_bandwidth=half_bandwidth)
        gf_mat = gfmatrix.construct_gf_omega(rv_inv=rv_inv, diag_inv=h_bar, rv=rv)
        gf_diag[:, i] = np.diag(gf_mat)
    return gf_diag


def get_gf_0_loc_deprecated(omega, params=None):
    r"""Old version, only diagonalizing in \epsilon.

    Might be necessary later to include self-energy.
    """
    # TODO: implement option do give back only unique layers
    prm = params
    diag = np.diag_indices_from(prm.t_mat)
    gf_0_inv_up = np.array(prm.t_mat, dtype=np.complex256, copy=True)
    gf_0_inv_up[diag] += prm.onsite_energy(sigma=sigma.up)

    gf_0_inv_dn = np.array(prm.t_mat, dtype=np.complex256, copy=True)
    gf_0_inv_dn[diag] += prm.onsite_energy(sigma=sigma.dn)

    gf_diag_up = invert_gf_0(omega, gf_0_inv_up, half_bandwidth=prm.D)
    gf_diag_dn = invert_gf_0(omega, gf_0_inv_dn, half_bandwidth=prm.D)
    return gf_diag_up, gf_diag_dn


def self_consistency(parameter, accuracy, mixing=1e-2, n_max=int(1e4)):
    """Naive self-consistency loop using simple mixing."""
    params = parameter
    iw_array = gf.matsubara_frequencies(np.arange(int(2**10)), beta=params.beta)

    # start loop paramters
    i, n, n_old = 0, 0, np.infty
    while np.linalg.norm(n - n_old) > accuracy:
        print('**** difference *****')
        print(np.linalg.norm(n - n_old))
        n_old = n
        gf_iw = params.gf0(iw_array)
        n = gf.density(gf_iw, potential=params.onsite_energy(), beta=params.beta)
        print('<n>: ', n)
        V_l = get_V(n.up + n.dn - np.average(n))  # FIXME: better dimension checks!!!
        print('V_l: ', V_l)
        params.V[:] = mixing * V_l + (1-mixing)*params.V
        print('V_l mixed: ', params.V[:])
        print(i)
        i += 1
        if i > n_max:
            print('maximum reached')
            break
    print('Final occupation:')
    print(n)
    print('Final potential')
    print(V_l)
    print(prm.V)
    print(np.linalg.norm(prm.V - V_l))


def update_occupation(n_start, i_omega, params):
    r"""Calculate new occupation by setting :math:`V_l` from the method `get_V`.

    This methods modifies `params.V`.

    Parameters
    ----------
    n_start : SpinResolvedArray(float, float)
        Start occupation used to calculate new occupation. The expected shape
        is (#spins=2, #layers).
    i_omega : array(complex)
        Matsubara frequencies :math:`iω_n` at which the Green's function is
        evaluated to calculate the occupation.
    params : prm
        `prm` object with the parameters set, determining the non-interacting
        Green's function. `params.V` will be updated.

    Returns
    -------
    update_occupation : array(float, float)
        The change in occupation caused by the potential obtained via `get_V`.
        Has the same shape as `n_start`.

    """
    assert n_start.shape[0] == 2
    params.V[:] = get_V(n_start.total - np.average(n_start.total))
    update_occupation.check_V.append(params.V.copy())
    gf_iw = params.gf0(i_omega)
    n = gf.density(gf_iw, potential=params.onsite_energy(), beta=params.beta)
    return n - n_start


def update_potential(V_start, i_omega, params):
    r"""Calculate new potential by setting :math:`V_l` from the method `get_V`.

    This methods modifies `params.V`.

    Parameters
    ----------
    V_start : array(float)
            Starting potential used to calculate the occupation and
            subsequently the new potential.
    i_omega : array(complex)
        Matsubara frequencies :math:`iω_n` at which the Green's function is
        evaluated to calculate the occupation.
    params : prm
        `prm` class with the parameters set, determining the non-interacting
        Green's function. `params.V` will be updated.

    Returns
    -------
    update_potential : array(float)
        The new occupation incorporating the potential obtained via `get_V`

    """
    params.V[:] = V_start
    gf_iw = params.gf0(i_omega)
    n = gf.density(gf_iw, potential=params.onsite_energy(), beta=params.beta)
    update_potential.check_n.append(n.copy())
    V = get_V(n.total - np.average(n.total))
    return V - V_start


def print_status(x, dx):
    """Intermediate print output for `optimize.root`."""
    print_status.itr += 1
    print("======== " + str(print_status.itr) + " =============")
    # print(x)
    print("--- " + str(np.linalg.norm(dx)) + " ---")


print_status.itr = 0


def broyden_self_consistency(parameters, accuracy, guess=None, kind='n'):
    """Charge self-consistency using root-finding algorithm.

    Parameters
    ----------
    parameters : prm
        `prm` object with the parameters set, determining Hamiltonian.
    accuracy : float
        Target accuracy of the self-consistency. Iteration stops if it is
        achieved.
    guess : ndarray, optional
        Starting value for the occupation or electrical potential.
    kind : {'n', 'V'}
        Weather self-consistency is determined according to the charge 'n' or
        the electrical potential 'V'. In the magnetic case 'V' seems to
        convergence better, there are half as many degrees of freedom.

    Returns
    -------
    n_opt : OptimizedResult
        The solution of the optimization routine, see `optimize.root`.
        `x` contains the actual values, and `success` states if convergence was
        achieved.

    """
    # TODO: better to determine V_l self-consistently? We have a start guess
    #       for V_l. On the other hand accuracy can easily be set from the
    #       accuracy for `n_l` determined by CT-Cyb
    #   -> Optimization of V_l takes for the magnetic case only have as many
    #      steps, as we only have have the parameters
    params = parameters
    iw_array = gf.matsubara_frequencies(np.arange(int(2**10)), beta=params.beta)
    if kind == 'n':
        if guess is None:
            gf_iw = params.gf0(iw_array)
            n_initial = gf.density(gf_iw, params.onsite_energy(),
                                   beta=params.beta)
        else:
            n_initial = guess
    elif kind == 'V':
        if guess is None:
            guess = np.zeros_like(parameters.mu)
        params.V[:] = guess
    # use a partial instead of args?
    print('======== optimize =======')
    n_opt = optimize.root(fun=update_occupation if kind == 'n' else update_potential,
                          x0=n_initial if kind == 'n' else guess,
                          args=(iw_array, params),
                          method='broyden1',
                          # method='anderson',
                          # method='linearmixing',
                          # method='excitingmixing',
                          # method='df-sane',
                          tol=accuracy,
                          # options={'nit': 3},
                          callback=print_status)
    print("=====================")
    print(n_opt.success)
    print('Optimized paramter:')
    print(n_opt.x)
    print('Final potential')
    print(params.V)
    print('***************')
    print(n_opt.message)
    return n_opt


# def main():
#    """Function to test convergence"""
if __name__ == '__main__':
    # Setup
    N = layers.size

    prm.T = 0.01
    prm.D = 1.  # half-bandwidth
    prm.mu = np.zeros(N)  # with respect to half filling
    prm.mu[N//2] = 0.45
    prm.V = np.zeros(N)
    prm.h = np.zeros(N)
    prm.h[N//2] = 0.9
    prm.U = np.zeros(N)
    # prm.U[0] = 0.8

    t = 0.2
    prm.t_mat = np.zeros((N, N))
    diag, _ = np.diag_indices_from(prm.t_mat)
    sdiag = diag[:-1]
    prm.t_mat[sdiag+1, sdiag] = prm.t_mat[sdiag, sdiag+1] = t

    prm.assert_valid()
    # self_consistency(prm, accuracy=2e-3, n_max=30)
    update_occupation.check_V = []
    update_potential.check_n = []
    opt_param = broyden_self_consistency(prm, accuracy=2e-3, kind='V')
    # return update_occupation.check_V
