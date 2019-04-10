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
from typing import Tuple
from collections import namedtuple

import numpy as np
import gftools as gt
from scipy.interpolate import UnivariateSpline

from . import __version__, charge, dataio
from .model import Hubbard_Parameters
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


def save_gf(gf_iw, self_iw, occ_layer, T, dir_='.', name='layer', compress=True):
    dataio.save_data(gf_iw=gf_iw, self_iw=self_iw, occ=occ_layer, temperature=T,
                     dir_=dir_, name=name, compress=compress)


def bare_iteration(it0, n_iter, gf_layer_iw0, self_layer_iw0, occ_layer0, function, **kwds):
    """Iterate the DMFT self-consistency equations.

    Parameters
    ----------
    it0 : int
        Starting number of the iterations. Needed for filenames.
    n_iter : int
        Number of iterations performed.
    gf_layer_iw0, self_layer_iw0 : (N_s, N_l, N_iw) complex np.ndarray
        Initial values for local Green's function and self energy. The shape of
        the arrays is (#spins=2, #layers, #Matsubara frequencies).
    occ_layer0 : (N_s, N_l) float np.ndarray
        Initial value for the occupation.
    function : callable
        The function implementing the self-consistency equations.
    kwds
        Additional keyword parameters passed to `function`

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


def sweep_update(prm: Hubbard_Parameters, iw_points, gf_layer_iw, self_layer_iw, occ_layer,
                 it, *, layer_config=None, data_T=None, n_process, **solver_kwds) -> LayerIterData:
    """Perform a sweep update, calculating the impurities for all layers.

    Parameters
    ----------
    prm : Hubbard_Parameters
        The model parameters.
    iw_points : (N_iw, ) complex np.ndarray
        The array of Matsubara frequencies.
    gf_layer_iw, self_layer_iw : (N_s, N_l, N_iw) complex np.ndarray
        The local Green's function and self-energy of the lattice.
    occ_layer : (N_s, N_l) float np.ndarray
        The local occupation of the *impurities* corresponding to the layers.
        The occupations have to match the self-energy (moments).
    it : int
        The iteration number needed for writing the files.
    layer_config : array_like of int, optional
        Mapping from the impurity models to the layers. For each layer an int
        is given which corresponds to the impurity model.
        E.g. for a symmetric setup of 4 layers `layer_config=(0, 1, 1, 0)`
        can be used to only solve 2 impurity models and symmetrically use the
        self energy for the related layers.
    data_T : float, optional
        The temperature corresponding to the input data `gf_layer_iw`, `self_layer_iw`
        and `occ_layer` if it differs from `prm.T`. This can be given, to interpolate
        the data to lower temperatures if `data_T > prm.T`.
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
    data_iw = iw_points if data_T is None \
        else gt.matsubara_frequencies(np.arange(iw_points.size), beta=1./data_T)
    siams = prm.get_impurity_models(
        z=data_iw, self_z=self_layer_iw, gf_z=gf_layer_iw, occ=occ_layer,
    )
    solve = partial(sb_qmc.solve, n_process=n_process, **solver_kwds)

    if layer_config:  # handle mapping layers -> SIAMs (symmetry considerations)
        layer_config = np.asarray(layer_config)
        layers = np.unique(layer_config)
        interacting_layers = layers[np.isin(layers, interacting_layers)]

    # handle temperature mismatch
    if data_T is not None and not np.allclose(prm.T, data_T):
        if data_T < prm.T:
            raise NotImplementedError(
                "Input data corresponds to lower temperatures than calculation.\n"
                "Only interpolation for larger temperatures implemented."
            )
        LOGGER.info("Input data temperature T=%s differs from calculation T=%s"
                    "\nHybridization functions will be interpolated.",
                    data_T, prm.T)
        interpolate_temperature = partial(interpolate, x_in=data_iw, x_out=iw_points)
    else:
        interpolate_temperature = None

    #
    # solve impurity model for the relevant layers
    #
    siam_iter = ((lay, siam) for lay, siam in enumerate(siams) if lay in interacting_layers)
    for lay, siam in siam_iter:
        LOGGER.progress('iter %s: starting layer %s with U = %s (%s)',
                        it, lay, siam.U, solver_kwds)
        if interpolate_temperature:
            LOGGER.progress("Interpolate hybridization fct (iter %s: lay %s)", it, lay)
            siam.hybrid_fct = interpolate_temperature(fct_in=siam.hybrid_fct)
            siam.z = iw_points
        data = solve(siam, output_name=f'iter{it}_lay{lay}')

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
            dir_=dataio.LAY_OUTPUT, name=f'layer_iter{it}')
    return LayerIterData(gf_iw=gf_layer_iw, self_iw=self_layer_iw, occ=occ_layer)


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
    """Execute DMFT loop."""
    warnings.warn("Currently not maintained! Will lack functionality but should work.")
    log_info(prm)

    # technical parameters
    N_IW = sb_qmc.N_IW

    # dependent parameters
    iw_points = gt.matsubara_frequencies(np.arange(N_IW), prm.beta)

    #
    # initial condition
    #
    (gf_layer_iw, self_layer_iw, occ_layer), start, __ = get_initial_condition(
        prm, kind=starting_point, iw_points=iw_points,
    )

    #
    # r-DMFT
    #
    # TODO: use numpy self-consistency mixing
    # iteration scheme
    converge = bare_iteration
    # sweep updates -> calculate all impurities, then update
    iteration = partial(sweep_update, prm, iw_points, n_process=n_process, **qmc_params)

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

        # dependent parameters
        iw_points = gt.matsubara_frequencies(np.arange(N_IW), prm.beta)

        #
        # initial condition
        #
        layerdat, start, data_T = get_initial_condition(
            prm, kind=starting_point, iw_points=iw_points, output_dir=output_dir,
        )

        # iteration scheme: sweep updates -> calculate all impurities, then update
        self.update: partial = partial(
            sweep_update, prm=prm, iw_points=iw_points,
            gf_layer_iw=layerdat.gf_iw, self_layer_iw=layerdat.self_iw, occ_layer=layerdat.occ,
            it=start
        )

        if not np.allclose(data_T, prm.T, atol=1e-14):
            # temperatures don't match
            if data_T < prm.T:
                raise NotImplementedError(
                    "Input data corresponds to lower temperatures than calculation.\n"
                    "Only interpolation for larger temperatures implemented."
                )
            self.update.keywords['data_T'] = data_T

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
        update_kdws.update(gf_layer_iw=data.gf_iw, self_layer_iw=data.self_iw, occ_layer=data.occ)

        if 'data_T' in update_kdws:
            # previously input data temperature didn't match prm.T
            del update_kdws['data_T']

        update_kdws['it'] += 1
        return data
