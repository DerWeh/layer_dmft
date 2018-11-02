"""r-DMFT loop."""
# encoding: utf-8
from pathlib import Path
from datetime import date

import numpy as np

import gftools as gt

from . import charge, model
from .model import prm
from .interface import sb_qmc

OUTPUT_DIR = "layer_output"


def save_gf(gf_iw, dir_='.', name='layer', compress=False):
    dir_ = Path(dir_).expanduser()
    dir_.mkdir(exist_ok=True)
    save_method = np.savez_compressed if compress else np.savez
    name = date.today().isoformat() + '_' + name
    save_method(dir_/name, gf_iw=gf_iw)


if __name__ == '__main__':
    # Paramters
    # ---------
    # layers = np.abs(np.arange(-19, 20))
    layers = np.abs(np.arange(-19, 20))

    N_l = layers.size

    # Hubbard model parameters
    prm.T = 0.01
    prm.D = 1.  # half-bandwidth
    prm.mu = np.zeros(N_l)  # with respect to half filling
    prm.mu[N_l//2] = 0.45
    prm.V = np.zeros(N_l)
    prm.h = np.zeros(N_l)
    prm.h[N_l//2] = 0.9
    prm.U = np.zeros(N_l)
    prm.U[N_l//2] = 0.8

    prm.hilbert_transform = model.hilbert_transform['bethe']

    t = 0.2
    prm.t_mat = np.zeros((N_l, N_l))
    diag, _ = np.diag_indices_from(prm.t_mat)
    sdiag = diag[:-1]
    prm.t_mat[sdiag+1, sdiag] = prm.t_mat[sdiag, sdiag+1] = t

    prm.assert_valid()

    # technical parameters
    N_IW = sb_qmc.N_IW
    ITER_MAX = 10

    # dependent parameters
    iw_points = gt.matsubara_frequencies(np.arange(N_IW), prm.beta)

    #
    # initial condition
    #
    print('start initialization')
    gf_layer_iw = prm.gf0(iw_points)  # start with non-interacting Gf
    if np.any(prm.U != 0):
        occ0 = prm.occ0(gf_layer_iw)
        tol = max(np.linalg.norm(occ0.err), 1e-14)
        charge.charge_self_consistency(prm, tol=tol, occ0=occ0.x, n_points=N_IW)
    print('done')

    # start with non-interacting solution
    self_layer_iw = np.zeros((2, N_l, N_IW), dtype=np.complex)

    #
    # r-DMFT
    #
    # sweep updates -> calculate all impurities, then update
    # TODO: use numpy self-consistency mixing
    for ii in range(ITER_MAX):
        for ll, U_l in enumerate(prm.U):
            if U_l == 0.:
                continue  # skip non-interacting, exact solution known
            # setup impurity solver
            sb_qmc.setup(prm, layer=ll, gf_iw=gf_layer_iw[:, ll], self_iw=self_layer_iw[:, ll])
            sb_qmc.run(n_process=2)
            sb_qmc.save_data(name=f'iter{ii}_lay{ll}')

            self_layer_iw[:, ll] = sb_qmc.read_self_energy_iw()
            print('finished', ll, U_l, flush=True)
        gf_layer_iw = prm.gf_dmft_s(iw_points, self_layer_iw)
        save_gf(gf_layer_iw, dir_=OUTPUT_DIR, name=f'layer_iter{ii}')
