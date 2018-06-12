import pytriqs as pt

from collections import namedtuple
from triqs_cthyb import Solver as Ct_hyb
from pytriqs.gf import (Gf, GfImFreq, GfImTime, GfReFreq, Idx, inverse,
                        iOmega_n, BlockGf)
from pytriqs.gf.meshes import MeshImFreq, MeshReFreq
from pytriqs.plot.mpl_interface import oplot

import numpy as np

import gftools as gf
import gftools.matrix as gfmatrix

_Spin = namedtuple('Spin', ['up', 'dn'])
spin = _Spin(up=0.5, dn=-0.5)


class _hubbard_model(type):
    def __repr__(self):
        _str = "Hubbard model parametr: "
        _str += ", ".join(('{}={}'.format(prm, getattr(self, prm))
                           for prm in self.__slots__))
        return _str


class prm(object):
    """Parameters of the Hubbard model."""
    __slots__ = ('T', 'D', 'mu', 'V', 'h', 'U', 't_mat')
    __metaclass__ = _hubbard_model

    @classmethod
    def onsite_energy(cls, spin):
        return cls.mu + 0.5*cls.U - cls.V - spin*cls.h


## Paramters
# layers = np.abs(np.arange(-19, 20))
version = 1
layers = np.abs(np.arange(-9, 10))

N = layers.size
labels, contract, expand = np.unique(layers, return_index=True, return_inverse=True)
contract_matrix = np.meshgrid(contract, contract)
Ni = contract.size


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
N_POINTS = 2**10  # number of Matsubara frequencies
ITER_MAX = 10

# dependent parameters
beta = 1./prm.T
iw_points = gf.matsubara_frequencies(np.arange(N_POINTS), beta)
interacting_labels = labels[prm.U!=0]


# check parameters
assert prm.mu.size == prm.h.size == prm.U.size == prm.V.size == Ni

G_inv_bare_up = np.asarray(prm.t_mat + np.diag(prm.onsite_energy(spin.up)[expand]), dtype=np.complex)
G_inv_bare_dn = np.asarray(prm.t_mat + np.diag(prm.onsite_energy(spin.dn)[expand]), dtype=np.complex)

mesh = MeshImFreq(beta, 'Fermion', N_POINTS)
gf_iw = BlockGf(mesh=mesh, gf_struct=[('up', list(labels)), ('dn', list(labels))],
                target_rank=2, name='Gf_layer_iw')


# FIXME: very inefficient, Self mostly empty data -> use target_shape
Self_iw_up = GfImFreq(mesh=mesh, indices=list(labels), name="Self_up")
# Self_iw_up = Gf(mesh=mesh, target_shape=[1], indices=[0], name="Self_up")
Self_iw_dn = GfImFreq(mesh=mesh, indices=list(labels), name="Self_dn")


# helper functions
# def get_g_bath_iw(g_loc_iw, label, sigma):
#     return inverse(iOmega_n + 0.5*U[i] + mu[i] - V[i] - sigma*h[i] - t*t*g_loc_iw)
def fill_gf(Gf_iw_up, Gf_iw_dn, Self_iw_up, Self_iw_dn):
    for n, iw in enumerate(iw_points):
        gf_inv_up = G_inv_bare_up.copy()
        # TODO: expand (check Self)
        gf_inv_up[diag, diag] += iw - Self_iw_up[Idx(n)][expand, expand]
        rv_inv, h, rv = gfmatrix.decompose_gf_omega(gf_inv_up)
        h_bar = gf.bethe_hilbert_transfrom(h, half_bandwidth=prm.D)
        gf_up = gfmatrix.construct_gf_omega(rv_inv, h_bar, rv)
        Gf_iw_up[Idx(n)] = gf_up[contract, contract]
        Gf_iw_up[Idx(-1-n)] = gf_up[contract, contract].conjugate()  # FIXME: adjoint? 
        # down
        gf_inv_dn = G_inv_bare_dn.copy()
        gf_inv_dn[diag, diag] += iw - Self_iw_dn[Idx(n)][expand, expand]
        rv_inv, h, rv = gfmatrix.decompose_gf_omega(gf_inv_dn)
        h_bar = gf.bethe_hilbert_transfrom(h, half_bandwidth=prm.D)
        gf_dn = gfmatrix.construct_gf_omega(rv_inv, h_bar, rv)
        Gf_iw_dn[Idx(n)] = gf_dn[contract_matrix]
        # FIXME: transpose is necessary
        Gf_iw_dn[Idx(-1-n)] = gf_dn[contract_matrix].conjugate()


interacting_layers = expand[prm.U[expand] != 0]


def fill_gf_imp(Gf_iw_up, Gf_iw_dn, gf_imp_iw_up, gf_imp_iw_dn):
    for n, iw in enumerate(iw_points):
        gf_inv_up = G_inv_bare_up.copy()
        # TODO: expand (check Self)
        gf_inv_up[diag, diag] += iw
        # FIXME: numerically unfavorable, change to [G+t^2*G*G_imp)/G_imp]?
        gf_inv_up[interacting_layers, interacting_layers] \
            = t*t*Gf_iw_up[Idx(n)][interacting_layers, interacting_layers] \
            + 1./gf_imp_iw_up[Idx(n)][interacting_layers, interacting_layers]
        rv_inv, h, rv = gfmatrix.decompose_gf_omega(gf_inv_up)
        h_bar = gf.bethe_hilbert_transfrom(h, half_bandwidth=prm.D)
        gf_up = gfmatrix.construct_gf_omega(rv_inv, h_bar, rv)
        Gf_iw_up[Idx(n)] = gf_up[contract, contract]
        Gf_iw_up[Idx(-1-n)] = gf_up[contract, contract].conjugate()
        # down
        gf_inv_dn = G_inv_bare_dn.copy()
        gf_inv_dn[diag, diag] += iw
        gf_inv_dn[interacting_layers, interacting_layers] \
            = t*t*Gf_iw_dn[Idx(n)][interacting_layers, interacting_layers] \
            + 1./gf_imp_iw_dn[Idx(n)][interacting_layers, interacting_layers]
        rv_inv, h, rv = gfmatrix.decompose_gf_omega(gf_inv_dn)
        h_bar = gf.bethe_hilbert_transfrom(h, half_bandwidth=prm.D)
        gf_dn = gfmatrix.construct_gf_omega(rv_inv, h_bar, rv)
        Gf_iw_dn[Idx(n)] = gf_dn[contract_matrix]
        # FIXME: transpose is necessary
        Gf_iw_dn[Idx(-1-n)] = gf_dn[contract_matrix].conjugate()


def plot_dos(gf_test):
    try:
        g_pade
    except NameError:
        g_pade = GfReFreq(target_shape=[], window=(-3., 3.), name="g_pade")
    g_pade.set_from_pade(gf_test, n_points=300)
    oplot(g_pade, mode='S')


# initial condition
Self_iw_up << 0.
Self_iw_dn << 0.
print 'start initialisation'
fill_gf(gf_iw['up'], gf_iw['dn'], Self_iw_up, Self_iw_dn)
print 'done'

# gf_test = Gf_iw_dn[0, 0]
# gf.density(gf_test.data[gf_test.data.size/2:], potential=+0.9, beta=beta)

ct_hyb = Ct_hyb(beta=beta, n_iw=N_POINTS,
                gf_struct=[('up', [0]), ('dn', [0])])


# # r-DMFT
# # for i in xrange(ITER_MAX):
if version == 1:
    for i, U_l in enumerate(prm.U):
        if U_l == 0.:
            continue
        ct_hyb.G0_iw['up'][0,0] << inverse(inverse(gf_iw['up'][i, i]) + Self_iw_up[i,i])
        ct_hyb.G0_iw['dn'][0,0] << inverse(inverse(gf_iw['dn'][i, i]) + Self_iw_dn[i,i])
        ct_hyb.solve(h_int=U_l*pt.operators.n('up',0)*pt.operators.n('dn',0),
                     n_cycles=100000, n_warmup_cycles=50000)
        # TODO: store values
        Self_iw_up[0,0] << ct_hyb.Sigma_iw['up'][0, 0]
        Self_iw_dn[0,0] << ct_hyb.Sigma_iw['dn'][0, 0]
        fill_gf(gf_iw['up'], gf_iw['dn'], Self_iw_up, Self_iw_dn)
        print 'finished', i, U_l
        break

# alternative for Bethe lattice:
if version == 2:
    g_imp_iw_up, g_imp_iw_dn = Self_iw_up.copy(), Self_iw_dn.copy()
    for i, U_l in enumerate(prm.U):
        cont_i = contract[i]
        if U_l == 0.:
            continue
        # !only valid for Bethe lattice
        # calculate explicitly -> neighboring layers untouched, in-plane removal
        ct_hyb.G0_iw['up'][0,0] << inverse(iOmega_n + G_inv_bare_up[cont_i, cont_i] - 0.25*prm.D*prm.D*gf_iw['up'][i, i])
        ct_hyb.G0_iw['dn'][0,0] << inverse(iOmega_n + G_inv_bare_dn[cont_i, cont_i] - t*t*gf_iw['dn'][i, i])
        ct_hyb.solve(h_int=U_l*pt.operators.n('up',0)*pt.operators.n('dn',0),
                     n_cycles=100000, n_warmup_cycles=50000)
        # TODO: store values, rename self
        g_imp_iw_up[0,0] << ct_hyb.G_iw['up'][0, 0]
        g_imp_iw_dn[0,0] << ct_hyb.G_iw['dn'][0, 0]
        fill_gf(gf_iw['up'], gf_iw['dn'], Self_iw_up, Self_iw_dn)
        # fill_gf_imp(Gf_iw_up, Gf_iw_dn, g_imp_iw_up, g_imp_iw_dn)
        print 'finished', i, U_l
        break
