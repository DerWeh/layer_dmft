"""r-DMFT loop.

Variables
---------
FORCE_PARAMAGNET: bool
    If `FORCE_PARAMAGNET` and no magnetic field, paramagnetism is enforce, i.e.
    the self-energy of ↑ and ↓ are set equal.

"""
# encoding: utf-8
import logging

from functools import partial
from typing import Tuple, Optional, Dict, Iterable, Any, NamedTuple
from collections import namedtuple

import numpy as np
import gftools as gt
from scipy.interpolate import UnivariateSpline

from . import __version__, charge, dataio, high_frequency_moments as hfm
from .model import Hubbard_Parameters, SIAM
from .interface import sb_qmc

# setup logging
LOGGER = logging.getLogger(__name__)
LOG_FMT = logging.Formatter('%(asctime)s|%(name)s|%(levelname)s: %(message)s')
# HANDLER = logging.StreamHandler()
HANDLER = logging.FileHandler("layer_dmft.log", mode='a')
HANDLER.setFormatter(LOG_FMT)
LOGGER.addHandler(HANDLER)

FORCE_PARAMAGNET = True


def log_info(prm: Hubbard_Parameters):
    """Log basic information for r-DMFT."""
    LOGGER.info("layer_dmft version: %s", __version__)
    LOGGER.info("gftools version:    %s", gt.__version__)
    LOGGER.info("%s", prm.pstr())


# DATA necessary for the DMFT iteration
LayerIterData = namedtuple('layer_iter_data', ['gf_iw', 'self_iw', 'occ'])
Sigma = namedtuple('sigma', ['iw', 'moments'])
SolverResult = NamedTuple("SolverResult", [('self', Sigma), ('data', Dict[str, Any])])


def save_gf(gf_iw, self_iw, occ_layer, T, dir_='.', name='layer', compress=True):
    dataio.save_data(gf_iw=gf_iw, self_iw=self_iw, occ=occ_layer, temperature=T,
                     dir_=dir_, name=name, compress=compress)


def interpolate_siam_temperature(siams: Iterable[SIAM], iw_n) -> Iterable[SIAM]:
    """Wrap interpolation of `SIAM.hybrid_tau` to continue different temperature."""
    interpolate_temperature: Optional[partial] = partial(interpolate, x_out=iw_n)
    for lay, siam in enumerate(siams):
        if siam.z[-1] < iw_n[-1]:
            # better message
            raise NotImplementedError(
                "Input data corresponds to lower temperatures than calculation.\n"
                "Only interpolation for larger temperatures implemented."
            )
        LOGGER.progress("Interpolate hybridization fct (lay %s)", lay)
        siam.hybrid_fct = interpolate_temperature(x_in=siam.z, fct_in=siam.hybrid_fct)
        siam.z = iw_n
        yield siam


def sweep_update(prm: Hubbard_Parameters, siams: Iterable[SIAM], iw_points,
                 it, *, layer_config=None, n_process, **solver_kwds) -> LayerIterData:
    """Perform a sweep update, calculating the impurities for all layers.

    Parameters
    ----------
    prm : Hubbard_Parameters
        The model parameters.
    iw_points : (N_iw, ) complex np.ndarray
        The array of Matsubara frequencies.
    it : int
        The iteration number needed for writing the files.
    layer_config : array_like of int, optional
        Mapping from the impurity models to the layers. For each layer an int
        is given which corresponds to the impurity model.
        E.g. for a symmetric setup of 4 layers `layer_config=(0, 1, 1, 0)`
        can be used to only solve 2 impurity models and symmetrically use the
        self energy for the related layers.
    n_process : int
        The number of precesses used by the `sb_qmc` code.
    solver_kwds
        Parameters passed to the impurity solver, here `sb_qmc`.

    Returns
    -------
    data.gf_iw, data.self_iw : (N_s, N_l, N_iw) complex np.ndarry
        The updated local Green's function and self-energy.
    data.occ : (N_s, N_l, N_iw) float np.ndarray
        Occupation obtained from the impurity models. As long as the DMFT is
        not converged, this *not* necessarily matches the occupation obtained
        from `data.gf_iw`.

    """
    interacting_layers = np.flatnonzero(prm.U)
    map_lay2imp = np.arange(prm._N_l) if layer_config is None else np.asarray(layer_config)
    unique_layers, map_imp2lay = np.unique(map_lay2imp[np.isin(map_lay2imp, interacting_layers)],
                                           return_inverse=True)
    # need better error handling
    assert interacting_layers.size == map_imp2lay.size

    solve = partial(sb_qmc.solve, n_process=n_process, **solver_kwds)

    def _solve(siam: SIAM, lay: int) -> SolverResult:
        LOGGER.progress('iter %s: starting layer %s with U = %s (%s)',
                        it, lay, siam.U, solver_kwds)
        data = solve(siam, output_name=f'iter{it}_lay{lay}')
        occ = -data['gf_tau'][:, -1]
        sm0 = hfm.self_m0(siam.U, occ[::-1])
        sm1 = hfm.self_m1(siam.U, occ[::-1])
        return SolverResult(self=Sigma(iw=data['self_energy_iw'], moments=[sm0, sm1]),
                            data=data)

    self_layer_iw = np.zeros((2, prm._N_l, iw_points.size), dtype=np.complex)
    occ_imp = np.zeros((2, prm._N_l))
    #
    # solve impurity model for the relevant layers
    #
    siam_iter = ((lay, siam) for lay, siam in enumerate(siams) if lay in unique_layers)
    solutions = list(_solve(siam, lay) for lay, siam in siam_iter)

    if layer_config is not None:
        LOGGER.progress('Using calculated self-energies from %s on layers %s',
                        list(unique_layers), list(interacting_layers))

    for lay, imp in zip(interacting_layers, map_imp2lay):
        LOGGER.debug("Assigning impurity %s (from %s) to layer %s",
                     imp, unique_layers[imp], lay)
        self_layer_iw[:, lay] = solutions[imp].self.iw
        occ_imp[:, lay] = -solutions[imp].data['gf_tau'][:, -1]

    # average over spin if not magnetic
    if FORCE_PARAMAGNET and np.all(prm.h == 0):
        self_layer_iw = np.mean(self_layer_iw, axis=0, keepdims=True)
        occ_imp = np.mean(occ_imp, axis=0, keepdims=True)

    gf_layer_iw = prm.gf_dmft_s(iw_points, self_layer_iw)

    if interacting_layers.size < prm._N_l:
        # calculated density from Gf for non-interacting layers
        occ = prm.occ0(gf_layer_iw, hartree=occ_imp[::-1], return_err=False)
        for lay, imp in zip(interacting_layers, map_imp2lay):
            occ[:, lay] = occ_imp[:, lay]
    else:
        occ = occ_imp

    # TODO: also save error, version, ...
    save_gf(gf_layer_iw, self_layer_iw, occ, T=prm.T,
            dir_=dataio.LAY_OUTPUT, name=f'layer_iter{it}')
    return LayerIterData(gf_iw=gf_layer_iw, self_iw=self_layer_iw, occ=occ)


def load_last_iteration(output_dir=None) -> Tuple[LayerIterData, int, float]:
    """Load relevant data from last iteration in `output_dir`.

    Parameters
    ----------
    output_dir : str, optional
        Directory from which data is loaded (default: `dataio.LAY_OUTPUT`)

    Returns
    -------
    layer_data : LayerIterData
        Green's function, self-energy and occupation
    last_iter : int
        Number of the last iteration
    temperature : float
        Temperature corresponding to the layer_data

    """
    last_iter, last_output = dataio.get_last_iter(dataio.LAY_OUTPUT if output_dir is None
                                                  else output_dir)
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
    """Calculate the Hartree approximation of `prm` for the r-DMFT loop.

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
        # non-interacting solution
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
    """Calculate the Hubbard I approximation of `prm` for the r-DMFT loop.

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
    hubbard_I_solution : tuple
        The Green's function, self-energy and occupation.

    """
    N_l = prm.mu.size
    N_iw = iw_n.size
    gf_layer_iw = prm.gf0(iw_n)  # start with non-interacting Gf
    occ0 = prm.occ0(gf_layer_iw)
    if np.any(prm.U != 0):
        # Non-interacting/Hartree is typically no good starting value!
        occ0.x[:] = .5
        output: Dict[str, np.ndarray] = {}
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


def get_initial_condition(prm: Hubbard_Parameters, kind='auto', iw_points=None, output_dir=None
                          ) ->Tuple[LayerIterData, int, float]:
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
        Directory from which data is loaded if `kind == 'resume'`
        (default: `dataio.LAY_OUTPUT`)

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
    if kind == 'auto':
        try:
            dataio.get_last_iter(dataio.LAY_OUTPUT if output_dir is None else output_dir)
            kind = 'resume'
        except IOError:
            LOGGER.info('No previous iterations found')
            kind = 'hartree'

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


def main(prm: Hubbard_Parameters, n_iter, n_process=1,
         qmc_params=sb_qmc.DEFAULT_QMC_PARAMS, starting_point='auto'):
    """Execute DMFT loop.

    Legacy method, prefer using `Runner` directly.
    """
    runner = Runner(prm, starting_point=starting_point)
    for __ in range(n_iter):
        runner.iteration(n_process=n_process, **qmc_params)


class Runner:
    """Run r-DMFT loop using `Runner.iteration`."""

    def __init__(self, prm: Hubbard_Parameters, starting_point='auto', output_dir=None) -> None:
        """Create runner for the model `prm`.

        Parameters
        ----------
        prm : Hubbard_Parameters
            Parameters of the Hamiltonian.
        starting_point : {'auto', 'resume', 'Hartree', 'Hubbard-I'}, optional
            What kind of starting point is used. 'resume' loads previous iteration
            (layer data with largest iteration number). 'hartree' starts from the
            static Hartree self-energy. 'hubbard-I' starts from the Hubbard-I
            approximation, using the atomic self-energy. 'auto' tries 'resume' and
            falls back to 'hartree'
        output_dir : str, optional
            Directory from which data is loaded if `kind == 'resume'`
            (default: `dataio.LAY_OUTPUT`)

        """
        log_info(prm)

        # technical parameters
        N_IW = sb_qmc.N_IW

        #
        # initial condition
        #
        iw_points = gt.matsubara_frequencies(np.arange(N_IW), prm.beta)
        layerdat, start, data_T = get_initial_condition(
            prm, kind=starting_point, iw_points=iw_points, output_dir=output_dir,
        )
        iw_points = gt.matsubara_frequencies(np.arange(N_IW), 1./data_T)

        siams = prm.get_impurity_models(iw_points, self_z=layerdat.self_iw,
                                        gf_z=layerdat.gf_iw, occ=layerdat.occ)

        if not np.allclose(data_T, prm.T, atol=1e-14):
            # temperatures don't match
            if data_T < prm.T:
                raise NotImplementedError(
                    "Input data corresponds to lower temperatures than calculation.\n"
                    "Only interpolation for larger temperatures implemented."
                )
            LOGGER.info("Input data temperature T=%s differs from calculation T=%s"
                        "\nHybridization functions will be interpolated.",
                        data_T, prm.T)
            # iw_points = gt.matsubara_frequencies(np.arange(N_IW), prm.beta)
            siams = interpolate_siam_temperature(siams, prm.T)

        # iteration scheme: sweep updates -> calculate all impurities, then update
        self.update = partial(sweep_update, prm=prm, siams=siams, iw_points=iw_points,
                              it=start)
        self.get_impurity_models = partial(prm.get_impurity_models, z=iw_points)

    def iteration(self, n_process=1, layer_config=None, **qmc_params):
        r"""Perform a DMFT iteration.

        Parameters
        ----------
        see `sweep_update`

        Returns
        -------
        data : LayerIterData
            Green's function, self-energy and occupation of the iteration.

        """
        # perform self-consistency loop
        data = self.update(layer_config=layer_config, n_process=n_process, **qmc_params)

        update_kdws = self.update.keywords
        siams = self.get_impurity_models(self_z=data.self_iw, gf_z=data.gf_iw, occ=data.occ)
        update_kdws.update(siams=siams)
        update_kdws['it'] += 1
        return data
