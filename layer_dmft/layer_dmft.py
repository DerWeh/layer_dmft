"""r-DMFT loop."""
# encoding: utf-8
import warnings

from pathlib import Path
from datetime import date, datetime

import numpy as np

import gftools as gt

from . import charge, model
from .model import prm, Hubbard_Parameters
from .interface import sb_qmc

OUTPUT_DIR = "layer_output"
CONTINUE = True


def write_info(prm: Hubbard_Parameters):
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


def save_gf(gf_iw, self_iw, dir_='.', name='layer', compress=True):
    dir_ = Path(dir_).expanduser()
    dir_.mkdir(exist_ok=True)
    save_method = np.savez_compressed if compress else np.savez
    name = date.today().isoformat() + '_' + name
    save_method(dir_/name, gf_iw=gf_iw, self_iw=self_iw)


def get_last_iter(dir_) -> (int, Path):
    """Return number and the file of the output of last iteration."""
    iter_files = Path(dir_).glob('*_iter*.npz')

    def _get_iter(file_object) -> int:
        r"""Return iteration `it` number of file with the name "\*_iter{it}.ENDING"."""
        basename = Path(file_object).stem
        ending = basename.split('_iter')[-1]
        try:
            it = int(ending)
        except ValueError:
            warnings.warn(f"Skipping unprocessable file: {file_object.name}")
            return None
        return it

    iters = {_get_iter(file_): file_ for file_ in iter_files}
    last_iter = max(iters.keys() - {None})  # remove invalid item
    return last_iter, iters[last_iter]


def converge(it0, n_iter, gf_layer_iw0, self_layer_iw0, function: callable):
    """Abstract function as template for the DMFT self-consistency.

    Parameters
    ----------
    it0 : int
        Starting number of the iterations. Needed for filenames.
    n_iter : int
        Number of iterations performed.
    gf_layer_iw0, self_layer_iw0 : (N_s, N_l, N_iw) complex np.ndarray
        Initial value for local Green's function and self energy. The shape of
        the arrays is (#spins=2, #layers, #Matsubara frequencies).
    function : callable
        The function implementing the self-consistency equations.

    Returns
    -------
    gf_layer_iw, self_layer_iw : (N_s, N_l, N_iw) complex np.ndarray
        The result for local Green's function and self energy after `n_iter`
        iterations.

    """
    del it0, n_iter, gf_layer_iw0, self_layer_iw0, function
    raise NotImplementedError("Abstract function only!"
                              " Needs to be overwritten by implementation.")
    return NotImplemented, NotImplemented  # pylint: disable=unreachable


def bare_iteration(it0, n_iter, gf_layer_iw0, self_layer_iw0, function):
    """Iterate the DMFT self-consistency equations.

    This is an implementation of `converge`.

    Parameters
    ----------
    see `converge`

    Returns
    -------
    gf_layer_iw, self_layer_iw : (N_s, N_l, N_iw) complex np.ndarray
        The result for local Green's function and self energy after `n_iter`
        iterations.

    """
    gf_layer_iw, self_layer_iw = gf_layer_iw0, self_layer_iw0
    for ii in range(it0, n_iter+it0):
        gf_layer_iw, self_layer_iw = function(gf_layer_iw, self_layer_iw, ii)
        save_gf(gf_layer_iw, self_layer_iw, dir_=OUTPUT_DIR, name=f'layer_iter{ii}')
    return gf_layer_iw, self_layer_iw


def get_sweep_updater(prm: Hubbard_Parameters, iw_points, n_process) -> callable:
    """Return a `sweep_update` function, calculating the impurities for all layers.

    Parameters
    ----------
    prm : Hubbard_Parameters
        The model parameters.
    iw_points : (N_iw, ) complex np.ndarray
        The array of Matsubara frequencies.
    n_process : int
        The number of precesses used by the `sb_qmc` code.

    Returns
    -------
    sweep_updater : callable
        The updater function. Its signature is
        `sweep_update(gf_layer_iw, self_layer_iw, it) -> gf_layer_iw, self_layer_iw`.

    """
    def sweep_update(gf_layer_iw, self_layer_iw, it):
        """Perform a sweep update, calculating the impurities for all layers.

        Parameters
        ----------
        gf_layer_iw, self_layer_iw : (N_s, N_l, N_iw) complex np.ndarray
            The local Green's function and self-energy of the lattice.
        it : int
            The iteration number needed for writing the files.

        Returns
        -------
        gf_layer_iw, self_layer_iw : (N_s, N_l, N_iw) complex np.ndarry
            The updated local Green's function and self-energy.

        """
        for ll, U_l in enumerate(prm.U):
            if U_l == 0.:
                continue  # skip non-interacting, exact solution known
            # setup impurity solver
            sb_qmc.setup(prm, layer=ll, gf_iw=gf_layer_iw[:, ll], self_iw=self_layer_iw[:, ll])
            sb_qmc.run(n_process=n_process)
            sb_qmc.save_data(name=f'iter{it}_lay{ll}')

            self_layer_iw[:, ll] = sb_qmc.read_self_energy_iw()
            print(f"iter {it}: finished layer {ll} with U = {U_l}", flush=True)
        gf_layer_iw = prm.gf_dmft_s(iw_points, self_layer_iw)
        return gf_layer_iw, self_layer_iw

    return sweep_update


def main(prm: Hubbard_Parameters, n_iter, n_process=1, qmc_params=sb_qmc.QMC_PARAMS):
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
        last_iter, last_output = get_last_iter("layer_output")
        print(f"Starting from iteration {last_iter}: {last_output.name}")

        with np.load(last_output) as data:
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
    # TODO: use numpy self-consistency mixing
    # iteration scheme
    converge = bare_iteration
    # sweep updates -> calculate all impurities, then update
    iteration = get_sweep_updater(prm, iw_points=iw_points, n_process=n_process)

    # perform self-consistency loop
    converge(
        it0=start, n_iter=n_iter,
        gf_layer_iw0=gf_layer_iw, self_layer_iw0=self_layer_iw, function=iteration
    )


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
