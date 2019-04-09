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

from functools import partial
from typing import Tuple, Optional, Callable, Dict
from pathlib import Path
from weakref import finalize
from datetime import date
from collections import OrderedDict, namedtuple, defaultdict

import numpy as np
import gftools as gt
from scipy.interpolate import UnivariateSpline

from . import __version__, charge
from .model import Hubbard_Parameters
from .interface import sb_qmc

# setup logging
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


def save_gf(gf_iw, self_iw, occ_layer, T, dir_='.', name='layer', compress=True):
    dir_ = Path(dir_).expanduser()
    dir_.mkdir(exist_ok=True)
    save_method = np.savez_compressed if compress else np.savez
    name = date.today().isoformat() + '_' + name
    save_method(dir_/name, gf_iw=gf_iw, self_iw=self_iw, occ=occ_layer, temperature=T)


def _get_iter(file_object) -> Optional[int]:
    r"""Return iteration `it` number of file with the name '\*_iter{it}(_*)?.ENDING'."""
    return _get_anystring(file_object, name='iter')


def _get_layer(file_object) -> Optional[int]:
    r"""Return iteration `it` number of file with the name '\*_lay{it}(_*)?.ENDING'."""
    return _get_anystring(file_object, name='lay')


def _get_anystring(file_object, name: str) -> Optional[int]:
    r"""Return iteration `it` number of file with the name '\*_{`name`}{it}(_*)?.ENDING'."""
    basename = Path(file_object).stem
    ending = basename.split(f'_{name}')[-1]  # select part after '_iter'
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


def get_last_iter(dir_) -> Tuple[int, Path]:
    """Return number and the file of the output of last iteration."""
    iter_files = Path(dir_).glob('*_iter*.npz')

    iters = {_get_iter(file_): file_ for file_ in iter_files}
    last_iter: int = max(iters.keys() - {None})  # remove invalid item
    return last_iter, iters[last_iter]


def get_all_iter(dir_) -> dict:
    """Return dictionary of files of the output with key `num`."""
    iter_files = Path(dir_).glob('*_iter*.npz')
    path_dict = {_get_iter(iter_f): iter_f for iter_f in iter_files
                 if _get_iter(iter_f) is not None}
    return path_dict


def get_all_imp_iter(dir_) -> Dict[int, Dict]:
    """Return directory of {int(layer): output} with key `num`."""
    iter_files = Path(dir_).glob('*_iter*_lay*.npz')
    path_dict: Dict[int, Dict] = defaultdict(dict)
    for iter_f in iter_files:
        it = _get_iter(iter_f)
        lay = _get_layer(iter_f)
        if (it is not None) and (lay is not None):
            path_dict[it][lay] = iter_f
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


class ImpurityData:
    """Interface to saved impurity data."""

    keys = {'gf_iw', 'gf_tau', 'self_iw', 'self_tau'}

    def __init__(self, dir_=sb_qmc.IMP_OUTPUT):
        """Mmap all data from directory."""
        self._filname_dict = get_all_imp_iter(dir_)
        mmap_dict = OrderedDict()
        for iter_key, iter_dict in sorted(self._filname_dict.items()):
            mmap_dict[iter_key] = OrderedDict(
                (key, self._autoclean_load(val, mmap_mode='r'))
                for key, val in sorted(iter_dict.items())
            )
        self.mmap_dict = mmap_dict
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

    @property
    def iterations(self):
        """Return list of iteration numbers."""
        return self.mmap_dict.keys()

    def __getitem__(self, key):
        """Emulate structured array behavior."""
        return self.mmap_dict[key]

    # def __getattr__(self, item):
    #     """Access elements in `keys`."""
    #     if item in self.keys:
    #         return self[item]
    #     raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")


def bare_iteration(it0, n_iter, gf_layer_iw0, self_layer_iw0, occ_layer0, function, **kwds):
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
        result = function(gf_layer_iw, self_layer_iw, occ_layer, it=ii, **kwds)
        gf_layer_iw, self_layer_iw, occ_layer = result
    return result


def get_sweep_updater(prm: Hubbard_Parameters, iw_points, n_process, **solver_kwds) -> Callable:
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
    def sweep_update(gf_layer_iw, self_layer_iw, occ_layer, it, layer_config=None, data_T=None):
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
        interacting_layers = np.flatnonzero(prm.U)
        siam_z = iw_points if data_T is None \
            else gt.matsubara_frequencies(np.arange(iw_points.size), beta=1./data_T)
        if layer_config is None:
            # TODO: return iterator instead of Tuple?
            interacting_siams = prm.get_impurity_models(
                z=siam_z, self_z=self_layer_iw, gf_z=gf_layer_iw, occ=occ_layer,
                only_interacting=True
            )
        else:
            layer_config = np.asarray(layer_config)
            layers = np.unique(layer_config)
            interacting_layers = layers[np.isin(layers, interacting_layers)]
            siams = prm.get_impurity_models(
                z=siam_z, self_z=self_layer_iw, gf_z=gf_layer_iw, occ=occ_layer,
                only_interacting=False
            )
            interacting_siams = (siams[lay] for lay in interacting_layers)

        #
        # solve impurity model for the relevant layers
        #
        for lay, siam in zip(interacting_layers, interacting_siams):
            LOGGER.progress('iter %s: starting layer %s with U = %s (%s)',
                            it, lay, siam.U, solver_kwds)
            if data_T is not None and not np.allclose(siam.T, data_T):
                if data_T > siam.T:
                    LOGGER.info("Interpolate hybridization fct (iter %s: lay %s) from T=%s to T=%s",
                                it, lay, data_T, siam.T)
                    siam.hybrid_fct = interpolate(siam.z, siam.hybrid_fct, iw_points)
                    siam.z = iw_points
                else:
                    raise NotImplementedError(
                        "Input data corresponds to lower temperatures than calculation.\n"
                        "Only interpolation for larger temperatures implemented."
                    )
            data = sb_qmc.solve(siam, n_process=n_process,
                                output_name=f'iter{it}_lay{lay}', **solver_kwds)

            self_layer_iw[:, lay] = data['self_energy_iw']
            occ_layer[:, lay] = -data['gf_tau'][:, -1]

        if layer_config is not None:
            LOGGER.progress('Using calculated self-energies on layers %s', list(layer_config))
            self_layer_iw = self_layer_iw[:, layer_config]

        if FORCE_PARAMAGNET and np.all(prm.h == 0):
            # TODO: think about using shape [1, N_l] arrays for paramagnet
            self_layer_iw[:] = np.mean(self_layer_iw, axis=0)

        gf_layer_iw = prm.gf_dmft_s(iw_points, self_layer_iw)
        # TODO: also save error
        save_gf(gf_layer_iw, self_layer_iw, occ_layer, T=prm.T,
                dir_=OUTPUT_DIR, name=f'layer_iter{it}')
        return LayerIterData(gf_iw=gf_layer_iw, self_iw=self_layer_iw, occ=occ_layer)

    return sweep_update


def load_last_iteration(output_dir=None) -> Tuple[LayerIterData, int, float]:
    """Load relevant data from last iteration in `output_dir` (def.: `OUTPUT_DIR`).

    Parameters
    ----------
    output_dir : str, optional
        Directory from which data is loaded (default: `OUTPUT_DIR`)

    Returns
    -------
    layer_data: tuple
        Green's function, self-energy and occupation
    last_iter: int
        Number of the last iteration

    """
    last_iter, last_output = get_last_iter(OUTPUT_DIR if output_dir is None else output_dir)
    LOGGER.info("Loading iteration %s: %s", last_iter, last_output.name)

    with np.load(last_output) as data:
        gf_layer_iw = data['gf_iw']
        self_layer_iw = data['self_iw']
        occ_layer = data['occ']
        temperature = data['temperature']
    result = LayerIterData(gf_iw=gf_layer_iw, self_iw=self_layer_iw, occ=occ_layer)
    return result, last_iter, temperature


@partial(np.vectorize, signature='(n),(n),(m)->(m)')
def interpolate(x_in, fct_in, x_out):
    """Calculate complex interpolation of `fct_in` and evaluate it at `x_out`.

    `x_in` and `x_out` can either be both real or imaginary, arbitrary contours
    are not supported.

    """
    if np.all(np.isreal(x_in)):
        x_out = x_out.real
    elif np.all(np.isreal(1j*x_in)):
        x_in = x_in.imag
        x_out = x_out.imag
    else:
        raise ValueError("Arbitrary complex numbers are not supported.\n"
                         "'x' has to be either real of imaginary.")
    smothening = len(x_in) * 1e-11
    Spline = partial(UnivariateSpline, s=smothening)
    fct_out = 1j*Spline(x_in, fct_in.imag)(x_out)
    fct_out += Spline(x_in, fct_in.real)(x_out)
    return fct_out


def hartree_solution(prm: Hubbard_Parameters, iw_n) -> LayerIterData:
    """Calculate the Hartree solution of `prm` for the r-DMFT loop.

    Parameters
    ----------
    prm : Hubbard_Parameters
        The parameters of the Hubbard model.
    iw_n : complex np.ndarray
        Matsubara frequencies. This determines the output shape,
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
        occ_layer = occ0.x
    return LayerIterData(gf_iw=gf_layer_iw, self_iw=self_layer_iw, occ=occ_layer)


def _hubbard_I_update(occ_init, i_omega, params: Hubbard_Parameters, out_dict):
    self_iw = gt.hubbard_I_self_z(i_omega, params.U, occ_init[::-1])
    out_dict['self_iw'] = self_iw = np.moveaxis(self_iw, 0, -1)
    out_dict['gf'] = gf_iw = params.gf_dmft_s(i_omega, self_z=self_iw, diagonal=True)
    occ = out_dict['occ'] = params.occ0(gf_iw, hartree=occ_init[::-1], return_err=False)
    return occ - occ_init


def hubbard_I_solution(prm: Hubbard_Parameters, iw_n) -> LayerIterData:
    N_l = prm.mu.size
    N_iw = iw_n.size
    gf_layer_iw = prm.gf0(iw_n)  # start with non-interacting Gf
    occ0 = prm.occ0(gf_layer_iw)
    if np.any(prm.U != 0):
        # Non-interacting/Hartree is typically no good starting value!
        occ0.x[:] = .5
        output = {}
        tol = max(np.linalg.norm(occ0.err), 1e-14)
        root_finder = charge._root
        optimizer = partial(_hubbard_I_update, i_omega=iw_n, params=prm, out_dict=output)
        solve = partial(root_finder, fun=optimizer, x0=occ0.x, tol=tol)
        solve()
        gf_layer_iw = output['gf']
        self_layer_iw = output['self_iw']
        occ_layer = output['occ']
    else:  # nonsense case
        # start with non-interacting solution
        self_layer_iw = np.zeros((2, N_l, N_iw), dtype=np.complex)
        occ_layer = occ0.x
    return LayerIterData(gf_iw=gf_layer_iw, self_iw=self_layer_iw, occ=occ_layer)


def get_initial_condition(prm: Hubbard_Parameters, kind='auto', iw_points=None, output_dir=None):
    """Get necessary quantities (G, Σ, n) to start DMFT loop.

    Parameters
    ----------
    prm : Hubbard_Parameters
        The Model parameters, necessary to calculate quantities.
    kind : {'auto', 'resume', 'Hartree', 'Hubbard-I'}, optional
        What kind of starting point is used. 'resume' loads previous iteration
        (layer data with largest iteration number). 'hartree' starts from the
        static Hartree self-energy. 'hubbard-I' starts from the Hubbard-I
        approximation, using the atomic self-energy. 'auto' tries 'resume' and
        falls back to 'hartree'
    iw_points : (N_iw,) complex np.ndarray, optional
        The Matsubara frequencies at which the quantities are calculated.
        Required if `kind` is not 'resume'.
    output_dir : str, optional
        Directory from which data is loaded if `kind == 'resume'` (default: `OUTPUT_DIR`)

    Returns
    -------
    layerdat.gf_iw : (2, N_l, N_iw) complex np.ndarray
        Diagonal part of lattice Matsubara Green's function.
    layerdat.self_iw : (2, N_l, N_iw) complex np.ndarray
        Local self-energy.
    layerdat.occ : (2, N_l) float np.ndarray
        Occupation
    start : int
        Number of first iteration. Number of loaded iteration plus one if 'resume',
        else `0`.

    """
    kind = kind.lower()
    assert kind in ('auto', 'resume', 'hartree', 'hubbard-i')
    # FIXME:  implement 'auto'
    if kind == 'resume':
        LOGGER.info("Reading old Green's function and self energy")
        layerdat, last_it, data_T = load_last_iteration(output_dir)
        start = last_it + 1
    elif kind == 'hartree':
        LOGGER.info('Start from Hartree approximation')
        layerdat = hartree_solution(prm, iw_n=iw_points)
        LOGGER.progress('DONE: calculated starting point')
        start = 0
        data_T = prm.T
    elif kind == 'hubbard-i':
        LOGGER.info('Start from Hubbard-I approximation')
        layerdat = hubbard_I_solution(prm, iw_n=iw_points)
        start = 0
        data_T = prm.T
    else:
        raise NotImplementedError('This should not have happened')

    return layerdat, start, data_T


# TODO: add resume=None -> "automatic choice"
def main(prm: Hubbard_Parameters, n_iter, n_process=1,
         qmc_params=sb_qmc.DEFAULT_QMC_PARAMS, resume=True):
    """Execute DMFT loop."""
    warnings.warn("Currently not maintained!")
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
        (gf_layer_iw, self_layer_iw, occ_layer), last_it, __ = load_last_iteration()
        start = last_it + 1
    else:
        LOGGER.info('Start from Hartree')
        gf_layer_iw, self_layer_iw, occ_layer = hartree_solution(prm, iw_n=iw_points)
        LOGGER.progress('DONE: calculated starting point')
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


class Runner:
    """Run r-DMFT loop using `Runner.iteration`."""

    def __init__(self, prm: Hubbard_Parameters, starting_point='auto', output_dir=None) -> None:
        """Create runner for the model `prm`.

        Parameters
        ----------
        prm : Hubbard_Parameters
            Parameters of the Hamiltonian.
        resume : bool
            If `resume`, load old r-DMFT data, else start from Hartree.
        output_dir : str, optional
            Directory from which data is loaded if `kind == 'resume'` (default: `OUTPUT_DIR`)

        """
        log_info(prm)
        self.prm = prm

        # technical parameters
        N_IW = sb_qmc.N_IW

        # dependent parameters
        self.iw_points = gt.matsubara_frequencies(np.arange(N_IW), prm.beta)

        #
        # initial condition
        #
        layerdat, start, data_T = get_initial_condition(
            prm, kind=starting_point, iw_points=self.iw_points, output_dir=output_dir,
        )
        self.iter_nr = start
        self.gf_iw = layerdat.gf_iw
        self.self_iw = layerdat.self_iw
        self.occ = layerdat.occ

        if not np.allclose(data_T, prm.T, atol=1e-14):
            if data_T < prm.T:
                raise NotImplementedError(
                    "Input data corresponds to lower temperatures than calculation.\n"
                    "Only interpolation for larger temperatures implemented."
                )
            LOGGER.info("Temperature of loaded data (%s) disagrees with parameters (%s)."
                        "Data will be interpolated.",
                        data_T, prm.T)
            self._temperature_missmatch = True
            self.data_T = data_T
        else:
            self._temperature_missmatch = False

        # iteration scheme
        self.converge = bare_iteration
        # sweep updates -> calculate all impurities, then update

    def iteration(self, n_process=1, layer_config=None, **qmc_params):
        r"""Perform a DMFT iteration.

        Parameters
        ----------
        n_process : int
            How many cores should be used.
        \*\*qmc_params :
            Parameters passed to the impurity solver, here `sb_qmc`.

        Returns
        -------
        data : LayerIterData
            Green's function, self-energy and occupation of the iteration.

        """
        iteration = get_sweep_updater(self.prm, iw_points=self.iw_points,
                                      n_process=n_process, **qmc_params)
        if self._temperature_missmatch:
            iteration = partial(iteration, data_T=self.data_T)
            # assuming no errors occur temperatures match as soon as function finishes
            self._temperature_missmatch = False

        # perform self-consistency loop
        data = self.converge(
            it0=self.iter_nr, n_iter=1,
            gf_layer_iw0=self.gf_iw, self_layer_iw0=self.self_iw, occ_layer0=self.occ,
            function=iteration,
            layer_config=layer_config
        )
        self.gf_iw, self.self_iw, self.occ = data
        self.iter_nr += 1
        return data
