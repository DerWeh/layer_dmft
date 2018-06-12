from builtins import (bytes, str, open, super, range,
                      zip, round, input, int, pow, object)
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


class SpinResolved(namedtuple('Spin', ['up', 'dn'])):
    __slots__ = ()

    def __getitem__(self, element):
        try:
            return super().__getitem__(element)
        except TypeError:
            return getattr(self, element)


spin = SpinResolved(up=0.5, dn=-0.5)


class _hubbard_model(type):
    def __repr__(self):
        _str = "Hubbard model parametr: "
        _str += ", ".join(('{}={}'.format(prm, getattr(self, prm))
                           for prm in self.__slots__))
        return _str


class prm(object):
    """Parameters of the Hubbard model.
    
    Attributes
    ----------
    T : float
        temperature
    D : float
        half-bandwidth
    mu : array(float)
        chemical potential of the layers
    V : array(float)
        electrical potential energy
    h : array(float)
        Zeeman magnetic field
    U : array(float)
        onsite interaction
    t_mat: array(float, float)
        hopping matrix
    """
    __slots__ = ('T', 'D', 'mu', 'V', 'h', 'U', 't_mat')
    __metaclass__ = _hubbard_model

    @classmethod
    def onsite_energy(cls, spin):
        return cls.mu + 0.5*cls.U - cls.V - spin*cls.h

    @classmethod
    def assert_valid(cls):
        """Raise error if attributes are not valid."""
        if not prm.mu.size == prm.h.size == prm.U.size == prm.V.size:
            raise ValueError(
                "all parameter arrays need to have the same shape"
                "mu: {cls.mu.size}, h: {cls.h.size}, U:{cls.U.size}, V: {cls.V.size}".format(cls)
            )


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
    up=np.asarray(prm.t_mat + np.diag(prm.onsite_energy(spin.up)[expand]), dtype=np.complex),
    dn=np.asarray(prm.t_mat + np.diag(prm.onsite_energy(spin.dn)[expand]), dtype=np.complex),
)

mesh = MeshImFreq(beta, 'Fermion', N_POINTS)
gf_iw = BlockGf(mesh=mesh, gf_struct=[('up', list(labels)), ('dn', list(labels))],
                target_rank=2, name='Gf_layer_iw')


# TODO: very inefficient, Self mostly empty data -> use target_shape
self_iw = BlockGf(mesh=mesh, gf_struct=[('up', list(labels)), ('dn', list(labels))],
                  target_rank=2, name='Sigma_layer_iw')
# Self_iw_up = Gf(mesh=mesh, target_shape=[1], indices=[0], name="Self_up")


# helper functions
# def get_g_bath_iw(g_loc_iw, label, sigma):
#     return inverse(iOmega_n + 0.5*U[i] + mu[i] - V[i] - sigma*h[i] - t*t*g_loc_iw)
def fill_gf(Gf_iw, Self_iw, g_inv_bare):
    for n, iw in enumerate(iw_points):
        gf_inv = g_inv_bare.copy()
        # TODO: expand (check Self)
        gf_inv[diag, diag] += iw - Self_iw[Idx(n)][expand, expand]
        rv_inv, h, rv = gfmatrix.decompose_gf_omega(gf_inv)
        h_bar = gf.bethe_hilbert_transfrom(h, half_bandwidth=prm.D)
        gf_up = gfmatrix.construct_gf_omega(rv_inv, h_bar, rv)
        Gf_iw[Idx(n)] = gf_up[contract, contract]
        # FIXME: transpose is necessary
        Gf_iw[Idx(-1-n)] = gf_up[contract, contract].conjugate()  # FIXME: adjoint?


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


# initial condition
self_iw << 0.
print 'start initialisation'
for sp in ('up', 'dn'):
    fill_gf(gf_iw[sp], self_iw[sp], g_inv_bare[sp])
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
        for sp in ('up', 'dn'):
            ct_hyb.G0_iw[sp][0,0] << inverse(inverse(gf_iw[sp][i, i]) + self_iw[sp][i, i])
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
