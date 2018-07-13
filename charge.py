# encoding: utf-8
"""Handles the charge self-consistency loop of the combined scheme."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from builtins import (ascii, bytes, chr, dict, filter, hex, input, int, map,
                      next, oct, open, pow, range, round, str, super, zip)
from functools import partial

import numpy as np

from scipy import optimize

import gftools as gf
import capacitor_formula
import gftools.matrix as gfmatrix

from model import prm, SpinResolvedArray, sigma

# V_DATA = 'loop/Vsteps.dat'
# DMFT_PARAM = 'layer_hb_param.init'
# OUTPUT = 'output.txt'


# FIXME
layers = np.arange(40)
# define get_V
e_schot = np.ones_like(layers) * 0.4
get_V = partial(capacitor_formula.potential_energy_vector,
                e_schot=e_schot, layer_labels=layers)


# def load_param():
#     """Return parameters if possible to load param file, else return None.
#
#     Assumes you are already in the proper directory.
#     """
#     with open(DMFT_PARAM, 'r') as param_file:
#         content_array = np.array(param_file.readlines(), dtype=object)
#     mask = [(not line.strip().startswith('#')) & (line.strip() != '')
#             for line in content_array]
#     bare_content = content_array[mask]  # striped all empty lines and comments

#     N_layer = int(bare_content[0])  # layers
#     NL = int(bare_content[1])  # labels
#     bare_mu_pos = 4 + N_layer + 8
#     hoppin_matrix = np.genfromtxt(bare_content[4:4+N_layer], dtype=int)
#     assert hoppin_matrix.shape == (N_layer, N_layer)
#     assert np.alltrue(hoppin_matrix.T == hoppin_matrix)
#     diagonal = np.diag(hoppin_matrix)
#     super_diag = np.diag(hoppin_matrix, k=1)
#     hopping_param = np.fromstring(bare_content[bare_mu_pos-2], sep=' ')

#     # parameters in order of file
#     # prm = type('Parameter', (), {})
#     prm = {}
#     prm["D"] = float(bare_content[bare_mu_pos - 4].strip())
#     prm["T"] = float(bare_content[bare_mu_pos - 3].strip())
#     # FIXME not very stable or readable
#     prm["t_nn"] = hopping_param[super_diag[NL-2:]]
#     prm["eps"] = hopping_param[diagonal[NL-1:]]
#     prm["U"] = np.fromstring(bare_content[bare_mu_pos - 1], sep=' ')
#     prm["mu"] = np.fromstring(bare_content[bare_mu_pos], sep=' ')
#     prm["h"] = np.fromstring(bare_content[bare_mu_pos + 1], sep=' ')
#     try:
#         prm["V"] = np.loadtxt(V_DATA, ndmin=2)[-1]
#     except IOError:
#         prm["V"] = np.zeros_like(prm['mu'])
#     else:
#         assert prm["V"].shape == prm["mu"].shape
#     return Parameter(N=NL, **prm)


# def get_impurity_lable():
#     """Get array of impurity label, can be used to expand arrays."""
#     with open(OUTPUT, 'r') as out_file:
#         for line in out_file.readlines():
#             if 'impurity label' in line:
#                 label_str = line.split("=")[1].strip()
#                 break
#         else:
#             raise EOFError("'impurity label' is not in {output}".format(output=OUTPUT))
#     return np.fromstring(label_str, dtype=int, sep=' ')


# def load_hopping_matrix():
#     with open(DMFT_PARAM, 'r') as param_file:
#         content_array = np.array(param_file.readlines(), dtype=object)
#     mask = [(not line.strip().startswith('#')) & (line.strip() != '')
#             for line in content_array]
#     bare_content = content_array[mask]  # striped all empty lines and comments

#     N_layer = int(bare_content[0])  # layers
#     bare_mu_pos = 4 + N_layer + 8
#     hoppin_matrix = np.genfromtxt(bare_content[4:4+N_layer], dtype=int)
#     assert hoppin_matrix.shape == (N_layer, N_layer)
#     assert np.alltrue(hoppin_matrix.T == hoppin_matrix)
#     hopping_param = np.fromstring(bare_content[bare_mu_pos-2], sep=' ')

#     t_mat = hopping_param[hoppin_matrix]
#     return t_mat


def invert_gf_0(omega, gf_0_inv, half_bandwidth):
    r"""Return the diagonal elements of the Green's function in real space.

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
    invert_gf_0 : array(complex)
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


def get_gf_0_loc(omega, params=None):
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


def get_gf_0_loc_opt(omega, params=None):
    # TODO: implement option do give back only unique layers
    prm = params
    diag = np.diag_indices_from(prm.t_mat)
    gf_0_inv_up = np.array(prm.t_mat, dtype=np.complex256, copy=True)
    gf_0_inv_up[diag] += prm.onsite_energy(sigma=sigma.up)
    rv_inv_up, xi_up, rv_up = gfmatrix.decompose_gf_omega(gf_0_inv_up)
    xi_bar_up = gf.bethe_hilbert_transfrom(omega[..., np.newaxis] + xi_up,
                                           half_bandwidth=prm.D)
    gf_up = np.einsum('ij, ...j, jk -> ...ik', rv_up, xi_bar_up, rv_inv_up)

    gf_0_inv_dn = np.array(prm.t_mat, dtype=np.complex256, copy=True)
    gf_0_inv_dn[diag] += prm.onsite_energy(sigma=sigma.dn)
    rv_inv_dn, xi_dn, rv_dn = gfmatrix.decompose_gf_omega(gf_0_inv_dn)
    xi_bar_dn = gf.bethe_hilbert_transfrom(omega[..., np.newaxis] + xi_dn,
                                           half_bandwidth=prm.D)
    gf_dn = np.einsum('ij, ...j, jk -> ...ik', rv_dn, xi_bar_dn, rv_inv_dn)

    return np.diagonal(gf_up, axis1=-2, axis2=-1), np.diagonal(gf_dn, axis1=-2, axis2=-1)


def occupation(gf_iw_local, params, spin):
    shape = gf_iw_local.shape
    gf_iw_local = np.reshape(gf_iw_local, (-1, shape[-1]))
    potential = prm.onsite_energy(sigma=spin)
    beta = 1./params.T
    occ = np.array([gf.density(gf_iw, potential=V, beta=beta) for gf_iw, V
                    in zip(gf_iw_local, potential)])
    return occ.reshape(shape[:-1])


def self_consistency(parameter, accuracy, mixing=1e-2, n_max=int(1e4)):
    params = parameter
    iw_array = gf.matsubara_frequencies(np.arange(int(2**10)), beta=1./params.T)

    # start loop paramters
    i, n, n_old = 0, 0, np.infty
    while np.linalg.norm(n - n_old) > accuracy:
        print('**** difference *****')
        print(np.linalg.norm(n - n_old))
        n_old = n
        gf_iw_up, gf_iw_dn = get_gf_0_loc(iw_array, params=params)
        n = SpinResolvedArray(up=occupation(gf_iw_up, params, sigma.up),
                              dn=occupation(gf_iw_dn, params, sigma.dn))
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
    n_start : array(float)
        Start occupation used to calculate new occupation.
    i_omega : array(complex)
        Matsubara frequencies :math:`iω_n` at which the Green's function is
        evaluated to calculate the occupation.
    params : prm
        `prm` class with the parameters set, determining the non-interacting
        Green's function. `params.V` will be updated.

    Returns
    -------
    update_occupation : array(float)
        The new occupation incorporating the potential obtained via `get_V`

    """
    assert n_start.shape[0] == 2
    params.V[:] = get_V(n_start[0] + n_start[1] - np.average(n_start[0] + n_start[1]))
    update_occupation.check_V.append(params.V.copy())
    gf_iw_up, gf_iw_dn = get_gf_0_loc(i_omega, params=params)
    n = SpinResolvedArray(up=occupation(gf_iw_up, params, spin=sigma.up),
                          dn=occupation(gf_iw_dn, params, spin=sigma.dn))
    return n - n_start


def update_potential(V_start, i_omega, params):
    # FIXME
    r"""Calculate new potential by setting :math:`V_l` from the method `get_V`.

    This methods modifies `params.V`.

    Parameters
    ----------
    n_start : array(float)
        Start occupation used to calculate new occupation.
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
    print("-Update V-")
    params.V[:] = V_start
    gf_iw_up, gf_iw_dn = get_gf_0_loc(i_omega, params=params)
    n = SpinResolvedArray(up=occupation(gf_iw_up, params, spin=sigma.up),
                          dn=occupation(gf_iw_dn, params, spin=sigma.dn))
    update_potential.check_n.append(n.copy())
    V = get_V(n.up + n.dn - np.average(n.up + n.dn))
    return V - V_start


def print_status(x, dx):
    print_status.itr += 1
    print("======== " + str(print_status.itr) + " =============")
    print(x)
    print("--- " + str(np.linalg.norm(dx)) + " ---")

print_status.itr = 0


def broyden_self_consistency(parameters, accuracy, guess=None, kind='n'):
    # TODO: better to determine V_l self-consistently? We have a start guess
    #       for V_l. On the other hand accuracy can easily be set from the
    #       accuracy for `n_l` determined by CT-Cyb
    if guess is None:
        guess = np.zeros_like(parameters.mu)
    params = parameters
    iw_array = gf.matsubara_frequencies(np.arange(int(2**10)), beta=1./params.T)
    gf_iw_up, gf_iw_dn = get_gf_0_loc(iw_array, params=params)
    n_initial = SpinResolvedArray(up=occupation(gf_iw_up, params, sigma.up),
                                  dn=occupation(gf_iw_dn, params, sigma.dn))
    # use a partial instead of args?
    print('======== start ==========')
    print(n_initial)
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


if __name__ == '__main__':
# def main():
    """Function to test convergence"""
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


# if __name__ == '__main__':
#     # params = load_param()
#     # iw_array = gf.matsubara_frequencies(np.arange(int(2**10)), beta=1./params.T)
#     # gf_iw_up, gf_iw_dn = get_gf_0_loc(iw_array)
#     # _, compres, expand = np.unique(get_impurity_lable(), return_index=True, return_inverse=True)
#     # print(occupation(gf_iw_dn[compres], params)[expand])
#     # print('finished')
#     # print('!!!!!!!!')
#     Vs = main()
