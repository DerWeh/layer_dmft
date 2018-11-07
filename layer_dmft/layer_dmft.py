"""r-DMFT loop."""
# encoding: utf-8
import warnings

from pathlib import Path
from datetime import date, datetime

import numpy as np

import gftools as gt

from . import charge, model
from .model import prm
from .interface import sb_qmc

OUTPUT_DIR = "layer_output"
CONTINUE = True


def write_info(prm: model.Hubbard_Parameters):
    """Write basic information for DMFT run to."""
    from ._version import get_versions

    with open('layer_output.txt', mode='a') as fp:
        fp.write("\n".join([
            datetime.now().isoformat(),
            "layer_dmft version: " + str(get_versions()['version']),
            "gftools version:    " + str(gt.__version__),
            "",
            prm.pstr(),
            "",
        ]))


def save_gf(gf_iw, self_iw, dir_='.', name='layer', compress=False):
    dir_ = Path(dir_).expanduser()
    dir_.mkdir(exist_ok=True)
    save_method = np.savez_compressed if compress else np.savez
    name = date.today().isoformat() + '_' + name
    save_method(dir_/name, gf_iw=gf_iw, self_iw=self_iw)


def main(prm: model.Hubbard_Parameters, n_iter, n_process=1, qmc_params=sb_qmc.QMC_PARAMS):
    """Execute DMFT loop."""
    write_info(prm)
    N_l = prm.mu.size

    # technical parameters
    N_IW = sb_qmc.N_IW

    # dependent parameters
    iw_points = gt.matsubara_frequencies(np.arange(N_IW), prm.beta)

    #
    # initial condition
    #
    if CONTINUE:
        print("reading old Green's function and self energy")
        iter_files = Path("layer_output").glob('*_iter*.npz')

        def _get_iter(file_objects):
            """Return iteration number of file with the name "\*_iter.ENDING"

            Parameters
            ----------
            file_objects : 
                file_objects is

            Returns
            -------
            _get_iter :

            """
            basename = file_objects.stem
            ending = basename.split('_iter')[-1]
            try:
                it = int(ending)
            except ValueError:
                warnings.warn(f"Skipping unprocessable file: {file_objects.name}")
                return None
            return it
        iters = {_get_iter(file_): file_ for file_ in iter_files}
        last_iter = max(iters.keys() - {None})  # remove invalid item
        print(f"Starting from iteration {last_iter}: {iters[last_iter].name}")

        with np.load(iters[last_iter]) as data:
            gf_layer_iw = data['gf_iw']
            self_layer_iw = data['self_iw']
        start = last_iter + 1
    else:
        print('start initialization')
        gf_layer_iw = prm.gf0(iw_points)  # start with non-interacting Gf
        if np.any(prm.U != 0):
            occ0 = prm.occ0(gf_layer_iw)
            tol = max(np.linalg.norm(occ0.err), 1e-14)
            opt_res = charge.charge_self_consistency(prm, tol=tol, occ0=occ0.x, n_points=N_IW)
            gf_layer_iw = prm.gf0(iw_points, hartree=opt_res.occ.x[::-1])
            self_layer_iw = np.zeros((2, N_l, N_IW), dtype=np.complex)
            self_layer_iw[:] = opt_res.occ.x[::-1, :, np.newaxis] * prm.U[np.newaxis, :, np.newaxis]
        else:  # nonsense case
            # start with non-interacting solution
            self_layer_iw = np.zeros((2, N_l, N_IW), dtype=np.complex)
        print('done')
        start = 0

    #
    # r-DMFT
    #
    # sweep updates -> calculate all impurities, then update
    # TODO: use numpy self-consistency mixing
    for ii in range(start, n_iter+start):
        for ll, U_l in enumerate(prm.U):
            if U_l == 0.:
                continue  # skip non-interacting, exact solution known
            # setup impurity solver
            sb_qmc.setup(prm, layer=ll, gf_iw=gf_layer_iw[:, ll], self_iw=self_layer_iw[:, ll])
            sb_qmc.run(n_process=n_process)
            sb_qmc.save_data(name=f'iter{ii}_lay{ll}')

            self_layer_iw[:, ll] = sb_qmc.read_self_energy_iw()
            print(f"iter {ii}: finished layer {ll} with U = {U_l}", flush=True)
        gf_layer_iw = prm.gf_dmft_s(iw_points, self_layer_iw)
        save_gf(gf_layer_iw, self_layer_iw, dir_=OUTPUT_DIR, name=f'layer_iter{ii}')


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
    prm.h[N_l//2] = -0.9
    prm.U = np.zeros(N_l)
    prm.U[N_l//2] = 0.8

    prm.hilbert_transform = model.hilbert_transform['bethe']

    t = 0.2
    prm.t_mat = np.zeros((N_l, N_l))
    diag, _ = np.diag_indices_from(prm.t_mat)
    sdiag = diag[:-1]
    prm.t_mat[sdiag+1, sdiag] = prm.t_mat[sdiag, sdiag+1] = t

    prm.assert_valid()
    ITER_MAX = 3
    qmc_params = sb_qmc.QMC_PARAMS
    main(prm, n_iter=ITER_MAX)
