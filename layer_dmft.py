# encoding: utf-8
from builtins import (bytes, str, open, super, range,
                      zip, round, input, int, pow, object)
import pytriqs as pt

from triqs_cthyb import Solver as Ct_hyb
from pytriqs.gf import BlockGf, Gf, GfImTime, GfReFreq, Idx, inverse, iOmega_n
from pytriqs.gf.meshes import MeshImFreq, MeshReFreq
from pytriqs.plot.mpl_interface import oplot

import numpy as np

import gftools as gf
import gftools.matrix as gfmatrix

from model import prm, sigma, SpinResolved, spins

# helper functions
# def get_g_bath_iw(g_loc_iw, label, sigma):
#     return inverse(iOmega_n + 0.5*U[i] + mu[i] - V[i] - sigma*h[i] - t*t*g_loc_iw)


def fill_gf(gf_iw, self_iw, g_inv_bare):
    ur"""Calculate data of `gf_iw` using the self-energy `self_iw`.

    The calculation uses the symmetry of `gf_iw`, no transpose is performed.
    Depends on the variables `expand` and `contract`.

    Parameters
    ----------
    gf_iw : BlockGf
        Green's function defined on a Matsubara frequency mesh.
        The data of this Green's function will be modified.
        The indices of `gf_iw` correspond to the labels of unique layers.
    self_iw : BlockGf
        Self-energy  defiend on a Matsubara frequency mesh.
        It is used to calculate the Green's function.
        The indices of `self_iw` correspond to the labels of unique layers.
    g_inv_bare : ndarray(complex)
        The inverse of the non-interacting Green's function stripped of the
        ϵ and ω dependence:
        :math:`G(iω, ϵ) - iω + ϵ`

    """
    for n, iw in enumerate(iw_points):
        gf_inv = g_inv_bare.copy()
        # TODO: expand (check Self)
        gf_inv[diag, diag] += iw - self_iw[Idx(n)][expand, expand]
        rv_inv, h, rv = gfmatrix.decompose_gf_omega(gf_inv)
        h_bar = gf.bethe_hilbert_transfrom(h, half_bandwidth=prm.D)
        gf_up = gfmatrix.construct_gf_omega(rv_inv, h_bar, rv)
        gf_iw[Idx(n)] = gf_up[contract, contract]
        # FIXME: transpose is necessary
        gf_iw[Idx(-1-n)] = gf_up[contract, contract].conjugate()  # FIXME: adjoint?


def fill_gf_imp(Gf_iw, gf_imp_iw, g_inv_bare):
    for n, iw in enumerate(iw_points):
        gf_inv_up = g_inv_bare.copy()
        # TODO: expand (check Self)
        gf_inv_up[diag, diag] += iw
        # FIXME: numerically unfavorable, change to [G+t^2*G*G_imp)/G_imp]?
        gf_inv_up[interacting_layers, interacting_layers] \
            = t*t*Gf_iw[Idx(n)][interacting_layers, interacting_layers] \
            + 1./gf_imp_iw[Idx(n)][interacting_layers, interacting_layers]
        rv_inv, h, rv = gfmatrix.decompose_gf_omega(gf_inv_up)
        h_bar = gf.bethe_hilbert_transfrom(h, half_bandwidth=prm.D)
        gf_up = gfmatrix.construct_gf_omega(rv_inv, h_bar, rv)
        Gf_iw[Idx(n)] = gf_up[contract, contract]
        # FIXME: transpose is necessary
        Gf_iw[Idx(-1-n)] = gf_up[contract, contract].conjugate()


def plot_dos(gf_test):
    try:
        g_pade
    except NameError:
        g_pade = GfReFreq(target_shape=[], window=(-3., 3.), name="g_pade")
    g_pade.set_from_pade(gf_test, n_points=300)
    oplot(g_pade, mode='S')


if __name__ == '__main__':
    # Paramters
    # ---------
    # layers = np.abs(np.arange(-19, 20))
    version = 1
    layers = np.abs(np.arange(-9, 10))

    N = layers.size
    labels, contract, expand = np.unique(layers, return_index=True, return_inverse=True)
    contract_matrix = np.meshgrid(contract, contract)  # mask to reduce the matrix
    Ni = contract.size

    # Hubbard model parameters
    prm.T = 0.01
    prm.D = 1.  # half-bandwidth
    prm.mu = np.zeros(Ni)  # with respect to half filling
    prm.mu[0] = 0.45
    prm.V = np.zeros(Ni)
    prm.h = np.zeros(Ni)
    prm.h[0] = 0.9
    prm.U = np.zeros(Ni)
    prm.U[0] = 0.8

    t = 0.2
    prm.t_mat = np.zeros((N, N))
    diag, _ = np.diag_indices_from(prm.t_mat)
    sdiag = diag[:-1]
    prm.t_mat[sdiag+1, sdiag] = prm.t_mat[sdiag, sdiag+1] = t

    # technical parameters
    # N_POINTS = 2**10  # number of Matsubara frequencies
    N_POINTS = 2**5
    ITER_MAX = 10
    ct_hyb_prm = {
        'n_cycles': 100000, 'n_warmup_cycles': 50000
    }

    # dependent parameters
    beta = 1./prm.T
    iw_points = gf.matsubara_frequencies(np.arange(N_POINTS), beta)
    interacting_labels = labels[prm.U != 0]
    interacting_layers = expand[prm.U[expand] != 0]

    # check parameters
    prm.assert_valid()

    g_inv_bare = SpinResolved(
        **{sp: np.asarray(prm.t_mat + np.diag(prm.onsite_energy(sigma[sp])[expand]), dtype=np.complex)
           for sp in spins}
    )

    mesh = MeshImFreq(beta, 'Fermion', N_POINTS)
    gf_iw = BlockGf(mesh=mesh, gf_struct=[(sp, list(labels)) for sp in spins],
                    target_rank=2, name='Gf_layer_iw')


    # TODO: very inefficient, Self mostly empty data -> use target_shape
    self_iw = BlockGf(mesh=mesh, gf_struct=[(sp, list(labels)) for sp in spins],
                      target_rank=2, name='Sigma_layer_iw')
    # Self_iw_up = Gf(mesh=mesh, target_shape=[1], indices=[0], name="Self_up")


    # initial condition
    self_iw << 0.
    print 'start initialisation'
    for sp in ('up', 'dn'):
        fill_gf(gf_iw[sp], self_iw[sp], g_inv_bare[sp])
    print 'done'

    # gf_test = Gf_iw_dn[0, 0]
    # gf.density(gf_test.data[gf_test.data.size/2:], potential=+0.9, beta=beta)

    ct_hyb = Ct_hyb(beta=beta, n_iw=N_POINTS,
                    gf_struct=[(sp, [0]) for sp in spins])

    # # r-DMFT
    # # for i in xrange(ITER_MAX):
    if version == 1:
        for i, U_l in enumerate(prm.U):
            if U_l == 0.:
                continue
            for sp in ('up', 'dn'):
                print(gf_iw[sp][i, i].data.shape)
                ct_hyb.G0_iw[sp][0,0] << inverse(inverse(gf_iw[sp][i, i]) + self_iw[sp][i, i])
                # ct_hyb.G0_iw[sp][0,0] << inverse(inverse(gf_iw[sp][i, i]) + self_iw[sp][i, i])
            oplot(ct_hyb.G0_iw.imag, '-o', label="general form")
            plt.legend()
            plt.ylim(ymax=0)
            plt.xlim(xmin=0)
            plt.show()
            break
            ct_hyb.solve(h_int=U_l*pt.operators.n('up',0)*pt.operators.n('dn',0),
                         **ct_hyb_prm)
            # TODO: store values
            for sp in ('up', 'dn'):
                self_iw[sp][i, i] << ct_hyb.Sigma_iw[sp][0, 0]
                fill_gf(gf_iw[sp], self_iw[sp], g_inv_bare[sp])
            print 'finished', i, U_l
            break

    # alternative for Bethe lattice:
    if version == 2:
        g_imp_iw = BlockGf(name_block_generator=self_iw, make_copies=True, name='Gf_imp_iw')
        for i, U_l in enumerate(prm.U):
            cont_i = contract[i]
            if U_l == 0.:
                continue
            # !only valid for Bethe lattice
            # calculate explicitly -> neighboring layers untouched, in-plane removal
            for sp in ('up', 'dn'):
                ct_hyb.G0_iw[sp][0,0] << inverse(iOmega_n + g_inv_bare[sp][cont_i, cont_i] - 0.25*prm.D*prm.D*gf_iw[sp][i, i])
            ct_hyb.solve(h_int=U_l*pt.operators.n('up',0)*pt.operators.n('dn',0),
                         **ct_hyb_prm)
            # TODO: store values, rename self
            for sp in ('up', 'dn'):
                g_imp_iw[sp][0,0] << ct_hyb.G_iw[sp][0, 0]
                # fill_gf(gf_iw[sp], self_iw[sp], g_inv_bare[sp])
                fill_gf_imp(gf_iw[sp], gf_iw[sp], g_inv_bare[sp])
            print 'finished', i, U_l
            break
