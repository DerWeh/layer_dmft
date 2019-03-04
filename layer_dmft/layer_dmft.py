"""r-DMFT loop.

Variables
---------
FORCE_PARAMAGNET: bool
    If `FORCE_PARAMAGNET` and no magnetic field, paramagnetism is enforce, i.e.
    the self-energy of ↑ and ↓ are set equal.

"""
# encoding: utf-8
import warnings
import logging

from typing import Tuple
from pathlib import Path
from weakref import finalize
from datetime import date
from collections import OrderedDict, namedtuple

import numpy as np
import gftools as gt

from . import __version__, charge
from .model import Hubbard_Parameters
from .interface import sb_qmc

# setup logging
PROGRESS = logging.INFO - 5
logging.addLevelName(PROGRESS, 'PROGRESS')
LOGGER = logging.getLogger(__name__)
LOG_FMT = logging.Formatter('%(asctime)s|%(name)s|%(levelname)s: %(message)s')
HANDLER = logging.StreamHandler()
# HANDLER = logging.FileHandler("layer_output.txt", mode='a')
HANDLER.setFormatter(LOG_FMT)
LOGGER.addHandler(HANDLER)
LOGGER.setLevel(PROGRESS)


OUTPUT_DIR = "layer_output"

FORCE_PARAMAGNET = True


def log_info(prm: Hubbard_Parameters):
    """Log basic information for r-DMFT."""
    LOGGER.info("layer_dmft version: %s", __version__)
    LOGGER.info("gftools version:    %s", gt.__version__)
    LOGGER.info("%s", prm.pstr())


# DATA necessary for the DMFT iteration
LayerIterData = namedtuple('layer_iter_data', ['gf_iw', 'self_iw', 'occ'])


def save_gf(gf_iw, self_iw, occ_layer, dir_='.', name='layer', compress=True):
    dir_ = Path(dir_).expanduser()
    dir_.mkdir(exist_ok=True)
    save_method = np.savez_compressed if compress else np.savez
    name = date.today().isoformat() + '_' + name
    save_method(dir_/name, gf_iw=gf_iw, self_iw=self_iw, occ=occ_layer)


def _get_iter(file_object) -> int:
    r"""Return iteration `it` number of file with the name '\*_iter{it}(_*)?.ENDING'."""
    basename = Path(file_object).stem
    ending = basename.split('_iter')[-1]  # select part after '_iter'
    iter_num = ending.split('_')[0]  # drop everything after possible '_'
    try:
        it = int(iter_num)
    except ValueError:
        warnings.warn(f"Skipping unprocessable file: {file_object.name}")
        return None
    return it


def get_iter(dir_, num) -> Path:
    """Return the file of the output of iteration `num`."""
    iter_files = Path(dir_).glob(f'*_iter{num}*.npz')

    paths = [iter_f for iter_f in iter_files if _get_iter(iter_f) == num]
    if not paths:
        raise AttributeError(f'Iterations {num} cannot be found.')
    if len(paths) > 1:
        raise AttributeError(f'Multiple occurrences of iteration {num}:\n'
                             + '\n'.join(str(element) for element in paths))
    return paths[0]


def get_last_iter(dir_) -> (int, Path):
    """Return number and the file of the output of last iteration."""
    iter_files = Path(dir_).glob('*_iter*.npz')

    iters = {_get_iter(file_): file_ for file_ in iter_files}
    last_iter = max(iters.keys() - {None})  # remove invalid item
    return last_iter, iters[last_iter]


def get_all_iter(dir_) -> dict:
    """Return dictionary of files of the output with key `num`."""
    iter_files = Path(dir_).glob('*_iter*.npz')
    path_dict = {_get_iter(iter_f): iter_f for iter_f in iter_files
                 if _get_iter(iter_f) is not None}
    return path_dict


class LayerData:
    """Interface to saved layer data."""

    keys = {'gf_iw', 'self_iw', 'occ'}

    def __init__(self, dir_=OUTPUT_DIR):
        """Mmap all data from directory."""
        self._filname_dict = get_all_iter(dir_)
        self.mmap_dict = OrderedDict((key, self._autoclean_load(val, mmap_mode='r'))
                                     for key, val in sorted(self._filname_dict.items()))
        self.array = np.array(self.mmap_dict.values(), dtype=object)

    def _autoclean_load(self, *args, **kwds):
        data = np.load(*args, **kwds)

        def _test():
            data.close()
        finalize(self, _test)
        return data

    def iter(self, it: int):
        """Return data of iteration `it`."""
        return self.mmap_dict[it]

    def iterations(self):
        """Return list of iteration numbers."""
        return self.mmap_dict.keys()

    def __getitem__(self, key):
        """Emulate structured array behavior."""
        try:
            return self.mmap_dict[key]
        except KeyError:
            if key in self.keys:
                return np.array([data[key] for data in self.mmap_dict.values()])
            else:
                raise

    def __getattr__(self, item):
        """Access elements in `keys`."""
        if item in self.keys:
            return self[item]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")


def abstract_converge(it0, n_iter, gf_layer_iw0, self_layer_iw0, function: callable):
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


def bare_iteration(it0, n_iter, gf_layer_iw0, self_layer_iw0, occ_layer0, function):
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
    gf_layer_iw, self_layer_iw, occ_layer = gf_layer_iw0, self_layer_iw0, occ_layer0
    for ii in range(it0, n_iter+it0):
        result = function(gf_layer_iw, self_layer_iw, occ_layer, it=ii)
        gf_layer_iw, self_layer_iw, occ_layer = result
        save_gf(gf_layer_iw, self_layer_iw, occ_layer,
                dir_=OUTPUT_DIR, name=f'layer_iter{ii}')
    return result


def get_sweep_updater(prm: Hubbard_Parameters, iw_points, n_process, **solver_kwds) -> callable:
    """Return a `sweep_update` function, calculating the impurities for all layers.

    Parameters
    ----------
    prm : Hubbard_Parameters
        The model parameters.
    iw_points : (N_iw, ) complex np.ndarray
        The array of Matsubara frequencies.
    n_process : int
        The number of precesses used by the `sb_qmc` code.
    solver_kwds:
        Parameters passed to the impurity solver, here `sb_qmc`.

    Returns
    -------
    sweep_updater : callable
        The updater function. Its signature is
        `sweep_update(gf_layer_iw, self_layer_iw, it) -> gf_layer_iw, self_layer_iw`.

    """
    def sweep_update(gf_layer_iw, self_layer_iw, occ_layer, it):
        """Perform a sweep update, calculating the impurities for all layers.

        Parameters
        ----------
        gf_layer_iw, self_layer_iw : (N_s, N_l, N_iw) complex np.ndarray
            The local Green's function and self-energy of the lattice.
        occ_layer : (N_s, N_l) float np.ndarray
            The local occupation of the *impurities* corresponding to the layers.
            The occupations have to match the self-energy (moments).
        it : int
            The iteration number needed for writing the files.

        Returns
        -------
        gf_layer_iw, self_layer_iw : (N_s, N_l, N_iw) complex np.ndarry
            The updated local Green's function and self-energy.

        """
        interacting_siams = prm.get_impurity_models(
            z=iw_points, self_z=self_layer_iw, gf_z=gf_layer_iw, occ=occ_layer,
            only_interacting=True
        )
        interacting_layers = np.flatnonzero(prm.U)

        for lay, siam in zip(interacting_layers, interacting_siams):
            LOGGER.log(PROGRESS, 'iter %s: starting layer %s with U = %s',
                       it, lay, siam.U)
            data = sb_qmc.solve(siam, n_process=n_process,
                                output_name=f'iter{it}_lay{lay}', **solver_kwds)

            self_layer_iw[:, lay] = data['self_energy_iw']
            occ_layer[:, lay] = -data['gf_tau'][:, -1]

        if FORCE_PARAMAGNET and np.all(prm.h == 0):
            # TODO: think about using shape [1, N_l] arrays for paramagnet
            self_layer_iw[:] = np.mean(self_layer_iw, axis=0)

        gf_layer_iw = prm.gf_dmft_s(iw_points, self_layer_iw)
        return LayerIterData(gf_iw=gf_layer_iw, self_iw=self_layer_iw, occ=occ_layer)

    return sweep_update


def load_last_iteration() -> Tuple[LayerIterData, int]:
    """Load relevant data from last iteration in `OUTPUT_DIR`.

    Returns
    -------
    layer_data: tuple
        Green's function, self-energy and occupation
    last_iter: int
        Number of the last iteration

    """
    last_iter, last_output = get_last_iter(OUTPUT_DIR)
    LOGGER.info("Loading iteration %s: %s", last_iter, last_output.name)

    with np.load(last_output) as data:
        gf_layer_iw = data['gf_iw']
        self_layer_iw = data['self_iw']
        occ_layer = data['occ']
    result = LayerIterData(gf_iw=gf_layer_iw, self_iw=self_layer_iw, occ=occ_layer)
    return result, last_iter


def hartree_solution(prm: Hubbard_Parameters, iw_n: int) -> LayerIterData:
    """Calculate the Hartree solution of `prm` for the r-DMFT loop.

    Parameters
    ----------
    prm : Hubbard_Parameters
        The parameters of the Hubbard model.
    iw_n : int
        Number of Matsubara frequencies. This determines the output shape,
        for how many points the Green's function and self-energy are calculated
        as well as the accuracy of the occupation which also enters the
        self-consistency for the Hartree solution.

    Returns
    -------
    hartree_solution : tuple
        The Green's function, self-energy and occupation.

    """
    N_l = prm.mu.size
    N_iw = iw_n.size
    gf_layer_iw = prm.gf0(iw_n)  # start with non-interacting Gf
    occ0 = prm.occ0(gf_layer_iw)
    if np.any(prm.U != 0):
        tol = max(np.linalg.norm(occ0.err), 1e-14)
        opt_res = charge.charge_self_consistency(prm, tol=tol, occ0=occ0.x, n_points=N_iw)
        gf_layer_iw = prm.gf0(iw_n, hartree=opt_res.occ.x[::-1])
        self_layer_iw = np.zeros((2, N_l, N_iw), dtype=np.complex)
        self_layer_iw[:] = opt_res.occ.x[::-1, :, np.newaxis] * prm.U[np.newaxis, :, np.newaxis]
        occ_layer = opt_res.occ.x
    else:  # nonsense case
        # start with non-interacting solution
        self_layer_iw = np.zeros((2, N_l, N_iw), dtype=np.complex)
        occ_layer = occ0
    return LayerIterData(gf_iw=gf_layer_iw, self_iw=self_layer_iw, occ=occ_layer)


def main(prm: Hubbard_Parameters, n_iter, n_process=1,
         qmc_params=sb_qmc.DEFAULT_QMC_PARAMS, resume=True):
    """Execute DMFT loop."""
    log_info(prm)

    # technical parameters
    N_IW = sb_qmc.N_IW

    # dependent parameters
    iw_points = gt.matsubara_frequencies(np.arange(N_IW), prm.beta)

    #
    # initial condition
    #
    if resume:
        LOGGER.info("Reading old Green's function and self energy")
        (gf_layer_iw, self_layer_iw, occ_layer), last_it = load_last_iteration()
        start = last_it + 1
    else:
        LOGGER.info('Start from Hartree')
        gf_layer_iw, self_layer_iw, occ_layer = hartree_solution(prm, iw_n=iw_points)
        LOGGER.log(PROGRESS, 'DONE: calculated starting point')
        start = 0

    #
    # r-DMFT
    #
    # TODO: use numpy self-consistency mixing
    # iteration scheme
    converge = bare_iteration
    # sweep updates -> calculate all impurities, then update
    iteration = get_sweep_updater(prm, iw_points=iw_points, n_process=n_process,
                                  **qmc_params)

    # perform self-consistency loop
    converge(
        it0=start, n_iter=n_iter,
        gf_layer_iw0=gf_layer_iw, self_layer_iw0=self_layer_iw, occ_layer0=occ_layer,
        function=iteration
    )
