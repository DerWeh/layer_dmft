"""r-DMFT loop.

Variables
---------
FORCE_PARAMAGNET: bool
    If `FORCE_PARAMAGNET` and no magnetic field, paramagnetism is enforce, i.e.
    the self-energy of ↑ and ↓ are set equal.

"""
# encoding: utf-8
import logging
import atexit

from collections import namedtuple
from functools import partial
from itertools import tee, chain, repeat
from typing import Optional, Dict, Iterable, Any, NamedTuple, Iterator

import numpy as np
import xarray as xr
from scipy.interpolate import UnivariateSpline
import gftools as gt

from . import charge, dataio, high_frequency_moments as hfm
from .model import Hubbard_Parameters, SIAM, Dim, matsubara_frequencies, rev_spin
from .interface import sb_qmc
from ._version import get_versions

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
    LOGGER.info("layer_dmft version: %s", get_versions()['version'])
    LOGGER.info("gftools version:    %s", gt.__version__)
    LOGGER.info("%s", prm.pstr())


# DATA necessary for the DMFT iteration
Sigma = namedtuple('sigma', ['iw', 'moments'])
SolverResult = NamedTuple("SolverResult", [('self', Sigma), ('occ', xr.DataArray), ('data', Dict[str, Any])])
MapLayer = namedtuple("MapLayer", ['interacting', 'unique', 'imp2lay', 'updated', 'unchanged'])


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


def mapping_lay_imp(prm_U, layer_config=None) -> MapLayer:
    """Handle custom mapping between layers and impurity models.

    Parameters
    ----------
    prm_U : (N_l, ) float np.ndarray
        Hubbard U of the model.
    layer_config : (N_l, ) int array_like, optional
        The custom mapping between layers and impurity models. Entries corresponding
        to *non-interacting* layers (U==0) will be *ignored*. For *interacting*
        layers, the number is the *index* of the layer from which the impurity
        model will be solved.
        If the number is *negative*, the results won't be changed (previous
        self-energy is reused).

    Returns
    -------
    mlayer.interacting : int np.ndarray
        Indices of interacting layers (U != 0)
    mlayer.unique : int np.ndarray
        Layers which will be mapped to a SIAM.
    mlayer.imp2lay : int np.ndarray
        Mapping for which layers each impurity model will be used.
    mlayer.updated : int np.ndarray
        Indices of layers, where new self-energy will be calculated.
    mlayer.unchanged : int np.ndarray
        Indices of layers, where the old self-energy will be reused.

    Raises
    ------
    ValueError
        If `layer_config` doesn't conform.

    Examples
    --------
    For a given model

    >>> prm = model.Hubbard_Parameters(5)
    >>> prm.U[:] = [0, 2, 2, 2, 0]
    >>> prm.t_mat = model.hopping_matrix(5, 1.)

    Values of `layer_config` for layers 0 and 4 will be ignored, as they are
    non-interacting. Layers 1 and 3 are equal due to symmetry, so we should
    map them to the same impurity model. Thus a meaningful choice would be

    >>> layer_config = [0, 1, 2, 1, 0]

    If the self energy for layer 2 is already accurate and we only want to
    improve layer 1 and 3, we can reuse the old result with

    >>> layer_config = [0, 1, -1, 1, 0]

    """
    N_l = len(prm_U)
    interacting_layers = np.flatnonzero(prm_U)
    map_lay2imp = np.arange(N_l) if layer_config is None else np.asarray(layer_config, dtype=int)
    if len(map_lay2imp) != N_l:
        raise ValueError(f"'layer_config' has wrong number of elements ({map_lay2imp.size}), "
                         f"expected: {N_l}")
    if np.any(map_lay2imp > N_l - 1):
        raise ValueError("'layer_config' doesn't point to valid layers"
                         f" (max: {N_l-1}): {layer_config}")
    map_lay2imp_int = map_lay2imp[interacting_layers]
    unique_layers, map_imp2lay = np.unique(map_lay2imp_int, return_inverse=True)
    assert interacting_layers.size >= map_imp2lay.size, \
        "There have to be more interacting layers than impurities"
    unchanged = interacting_layers[np.flatnonzero(map_lay2imp_int < 0)]
    updated = interacting_layers[np.flatnonzero(map_lay2imp_int >= 0)]
    assert updated.size == map_imp2lay.size, \
        "Every updated layer must be mapped to an impurity model"
    return MapLayer(interacting_layers, unique_layers, map_imp2lay, updated, unchanged)


def sweep_update(prm: Hubbard_Parameters, siams: Iterable[SIAM], iw_points,
                 it, *, layer_config=None, self_iw=None, occ=None,
                 n_process, solve=sb_qmc.solve, **solver_kwds) -> xr.Dataset:
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
    mlayer = mapping_lay_imp(prm.U, layer_config)

    solve = partial(solve, n_process=n_process, **solver_kwds)

    def _solve(siam: SIAM, lay: int) -> SolverResult:
        LOGGER.progress('iter %s: starting layer %s with U = %s (%s)',
                        it, lay, siam.U, solver_kwds)
        data = solve(siam, output_name=f'iter{it}_lay{lay}')
        _occ = xr.DataArray(-data['gf_tau'][:, -1], dims=[Dim.sp], coords=[['up', 'dn']],
                            attrs={'layer': lay})
        sm0 = hfm.self_m0(siam.U, rev_spin(_occ))
        sm1 = hfm.self_m1(siam.U, rev_spin(_occ))
        return SolverResult(self=Sigma(iw=data['self_energy_iw'], moments=[sm0, sm1]),
                            occ=_occ, data=data)

    self_layer_iw = xr.DataArray(
        np.zeros((2, prm.N_l, iw_points.size), dtype=np.complex),
        name='Σ', dims=[Dim.sp, Dim.lay, Dim.iws],
        coords={Dim.sp: ['up', 'dn'], Dim.lay: range(prm.N_l), Dim.iws: iw_points.coords[Dim.iws].values}
    )
    occ_imp = xr.DataArray(np.zeros((2, prm.N_l)), dims=[Dim.sp, Dim.lay], name='occ',
                           coords=[['up', 'dn'], range(prm.N_l)])
    #
    # solve impurity model for the relevant layers
    #
    siam_iter = ((lay, siam) for lay, siam in enumerate(siams) if lay in mlayer.unique)
    solutions = list(_solve(siam, lay) for lay, siam in siam_iter)

    if layer_config is not None:
        LOGGER.progress('Using calculated self-energies from %s on layers %s',
                        list(mlayer.unique), list(mlayer.imp2lay))

    for lay, imp in zip(mlayer.updated, mlayer.imp2lay):
        LOGGER.debug("Assigning impurity %s (from %s) to layer %s",
                     imp, mlayer.unique[imp], lay)
        self_layer_iw[{Dim.lay: lay}] = solutions[imp].self.iw
        occ_imp[{Dim.lay: lay}] = -solutions[imp].occ
    for lay in mlayer.unchanged:
        LOGGER.debug("Reusing old values for layer %s", lay)
        self_layer_iw[{Dim.lay: lay}] = self_iw[{Dim.lay: lay}]
        occ_imp[{Dim.lay: lay}] = occ[{Dim.lay: lay}]

    # average over spin if not magnetic
    if FORCE_PARAMAGNET and np.all(prm.h == 0):
        self_layer_iw = self_layer_iw.mean(dim=Dim.sp, keep_attrs=True, keepdims=True)
        occ_imp = occ_imp.mean(dim=Dim.sp, keep_attrs=True, keepdims=True)

    gf_layer_iw = prm.gf_dmft_s(iw_points, self_layer_iw)

    if mlayer.interacting.size < prm.N_l:
        # calculated density from Gf for non-interacting layers
        occ = prm.occ0(gf_layer_iw, hartree=rev_spin(occ_imp), return_err=False)
        occ[{Dim.lay: mlayer.interacting}] = occ_imp[{Dim.lay: mlayer.interacting}]
    else:
        occ = occ_imp

    data = xr.Dataset(
        {'gf_iw': gf_layer_iw, 'self_iw': self_layer_iw, 'occ': occ},  # , 'onsite-paramters': prm.params},
        attrs={'temperature': prm.T,
               '__version__': get_versions()['version'], 'gftools.__version__': gt.__version__,
               Dim.it: it}
    )
    dataio.save_dataset(data, dir_=dataio.LAY_OUTPUT, name=f'layer_iter{it}')
    return data


def load_last_iteration(output_dir=None) -> xr.Dataset:
    """Load relevant data from last iteration in `output_dir`.

    Parameters
    ----------
    output_dir : str, optional
        Directory from which data is loaded (default: `dataio.LAY_OUTPUT`)

    Returns
    -------
    layer_data : xr.Dataset
        Green's function, self-energy and occupation, iteration number and
        temperature

    """
    last_iter, last_output = dataio.get_last_iter(dataio.LAY_OUTPUT if output_dir is None
                                                  else output_dir)
    LOGGER.info("Loading iteration %s: %s", last_iter, last_output.name)

    result = xr.open_dataset(last_output, engine='h5netcdf')
    return result


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


def hartree_solution(prm: Hubbard_Parameters, iw_n) -> xr.Dataset:
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
    gf_iw = prm.gf0(iw_n)  # start with non-interacting Gf
    occ0 = prm.occ0(gf_iw)
    if np.all(prm.U == 0):  # nonsense case, no need for Hartree
        return xr.Dataset({'gf_iw': gf_iw, 'self_iw': xr.zeros_like(gf_iw), 'occ': occ0.x})
    tol = max(np.linalg.norm(occ0.err), 1e-14)
    opt_res = charge.charge_self_consistency(prm, tol=tol, occ0=occ0.x, n_points=iw_n.size)
    gf_iw = prm.gf0(iw_n, hartree=opt_res.occ.x.roll({Dim.sp: 1}, roll_coords=False))
    self_iw = opt_res.occ.x.roll({Dim.sp: 1}, roll_coords=False) * prm.U
    self_iw = xr.broadcast(self_iw, gf_iw)[0].astype(np.complex)
    self_iw.name = 'Σ_{Hartree}'
    return xr.Dataset({'gf_iw': gf_iw, 'self_iw': self_iw, 'occ': opt_res.occ.x})


def _hubbard_I_update(occ_init, i_omega, params: Hubbard_Parameters, out_dict):
    # called by solver -> propagates only values
    self_iw = gt.hubbard_I_self_z(i_omega.values, params.U.values, occ_init[::-1])
    out_dict['self_iw'] = self_iw = np.moveaxis(self_iw, 0, -1)
    out_dict['gf'] = gf_iw = params.gf_dmft_s(i_omega, self_z=self_iw, diagonal=True)
    occ = out_dict['occ'] = params.occ0(gf_iw.values, hartree=occ_init[::-1], return_err=False)
    return occ.values - occ_init


def hubbard_I_solution(prm: Hubbard_Parameters, iw_n) -> xr.Dataset:
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
    gf_iw = prm.gf0(iw_n)  # start with non-interacting Gf
    occ0 = prm.occ0(gf_iw)
    if np.all(prm.U == 0):  # nonsense case
        return xr.Dataset({'gf_iw': gf_iw, 'self_iw': xr.zeros_like(gf_iw), 'occ': occ0.x})
    occ0.x[:] = .5  # Non-interacting/Hartree is typically no good starting value!
    output: Dict[str, np.ndarray] = {}
    tol = max(np.linalg.norm(occ0.err), 1e-14)
    root_finder = charge._root  # pylint: disable=protected-access
    optimizer = partial(_hubbard_I_update, i_omega=iw_n, params=prm, out_dict=output)
    root_finder(fun=optimizer, x0=occ0.x, tol=tol)
    self_iw = (output['gf'].dims, output['self_iw'])
    return xr.Dataset({'gf_iw': output['gf'], 'self_iw': self_iw, 'occ': output['occ']})


def get_initial_condition(prm: Hubbard_Parameters, kind='auto', iw_points=None, output_dir=None
                          ) -> xr.Dataset:
    """Get necessary quantities (G, Σ, n) to start DMFT loop.

    Parameters
    ----------
    prm : Hubbard_Parameters
        The Model parameters, necessary to calculate quantities.
    kind : {'auto', 'resume', 'Hartree', 'Hubbard-I', dict}, optional
        What kind of starting point is used. 'resume' loads previous iteration
        (layer data with largest iteration number). 'hartree' starts from the
        static Hartree self-energy. 'hubbard-I' starts from the Hubbard-I
        approximation, using the atomic self-energy. 'auto' tries 'resume' and
        falls back to 'hartree'.
        Alternatively a dict (or Dataset) can be given, to
        explicitly specify the self-energy (and optionally the occupation and
        Green's function).
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
    try:
        kind = kind.lower()
    except AttributeError:
        pass
    else:
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
        return load_last_iteration(output_dir)
    if kind == 'hartree':
        LOGGER.info('Start from Hartree approximation')
        layerdat = hartree_solution(prm, iw_n=iw_points)
        LOGGER.progress('DONE: calculated starting point')
    elif kind == 'hubbard-i':
        LOGGER.info('Start from Hubbard-I approximation')
        layerdat = hubbard_I_solution(prm, iw_n=iw_points)
        LOGGER.progress('DONE: calculated starting point')
    else:  # giving a xr.Dataset or a dict
        try:  # assuming kind is xr.Dataset
            kind = dict(kind.data_vars)
        except AttributeError:
            pass

        kind: Dict
        try:
            self_iw = kind.pop('self_iw')
        except KeyError:
            raise NotImplementedError('This should not have happened')
        gf_iw = kind.pop('gf_iw') if 'gf_iw' in kind else prm.gf_dmft_s(z=iw_points, self_z=self_iw)
        occ = kind.pop('occ') if 'occ' in kind else prm.occ0(gf_iw, return_err=False)
        if kind:
            raise TypeError("If `kind` is a dict, it may only have the keys 'self_iw'"
                            "and optionally 'gf_iw' or 'occ'.\n"
                            f"Other keys: {tuple(kind.keys())}")
        layerdat = xr.Dataset({'gf_iw': gf_iw, 'self_iw': self_iw, 'occ': occ})
    layerdat.attrs.update(**{Dim.it: -1, 'temperature': prm.T})
    return layerdat


def main(prm: Hubbard_Parameters, n_iter, n_process=1,
         qmc_params=sb_qmc.DEFAULT_QMC_PARAMS, starting_point='auto'):
    """Execute DMFT loop.

    Legacy method, prefer using `Runner` directly.
    """
    runner = Runner(prm, starting_point=starting_point)
    for __ in range(n_iter):
        runner.iteration(n_process=n_process, **qmc_params)


def mixed_siams(mixing: float, new: Iterable[SIAM], old: Iterable[SIAM]) -> Iterator[SIAM]:
    """Mix the hybridization function of `new` and `old` SIAMs.

    The `SIAM` objects from `old` will be consumed, while the new SIAMs
    remain unchanged.

    Parameters
    ----------
    mixing : float
        How much of the `new` hybridization function will be used, `1.` would
        mean only the new one, `0.` only the old one.
    new, old
        Iterables containing the new (old) SIAMs.

    Yields
    ------
    mixed_siam : SIAM
        The SIAM with the mixed hybridization function in *τ* space.
        No data in *iω* exists! This object can *only* be used to obtain
        onsite energies and hybridization function of *τ* Δ(τ).

    """
    assert 0. <= mixing <= 1.
    for siam_new, siam_old in zip(new, old):
        assert siam_old.T == siam_new.T
        assert siam_old.U == siam_new.U
        assert np.all(siam_old.e_onsite == siam_new.e_onsite)
        hyb_iw_mixed = mixing*siam_new.hybrid_fct + (1-mixing)*siam_old.hybrid_fct
        hyb_mom_mixed = mixing*siam_new.hybrid_mom + (1-mixing)*siam_old.hybrid_mom
        yield SIAM(e_onsite=siam_new.e_onsite, U=siam_old.U, T=siam_old.T,
                   z=siam_old.z, hybrid_fct=hyb_iw_mixed, hybrid_mom=hyb_mom_mixed)


class Runner:
    """Run r-DMFT loop using `Runner.iteration`."""

    def __init__(self, prm: Hubbard_Parameters, starting_point='auto', output_dir=None,
                 default_solver=sb_qmc.solve) -> None:
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
        default_solver : Callable
            Function which will be called to solve the impurity model, if not
            explicitly specified.

        """
        log_info(prm)
        self.default_solver = default_solver
        # technical parameters
        N_IW = sb_qmc.N_IW

        #
        # initial condition
        #
        iw_points = matsubara_frequencies(np.arange(N_IW), prm.beta)
        data = get_initial_condition(
            prm, kind=starting_point, iw_points=iw_points, output_dir=output_dir,
        )
        data_T = data.temperature
        iw_points = matsubara_frequencies(np.arange(N_IW), 1./data_T)

        siams = prm.get_impurity_models(iw_points, self_z=data.self_iw,
                                        gf_z=data.gf_iw, occ=data.occ)

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
            iw_points = matsubara_frequencies(np.arange(N_IW), prm.beta)
            siams = interpolate_siam_temperature(siams, iw_points)

        # iteration scheme: sweep updates -> calculate all impurities, then update
        self.update = partial(sweep_update, prm=prm, siams=siams, iw_points=iw_points,
                              it=data.attrs[Dim.it]+1, self_iw=data.self_iw, occ=data.occ)
        self.get_impurity_models = partial(prm.get_impurity_models, z=iw_points)
        atexit.register(next(_finished_message))

    def iteration(self, n_process=1, layer_config=None, solver=None, **qmc_params):
        r"""Perform a DMFT iteration.

        Parameters
        ----------
        see `sweep_update`
        layer_config
            see `mapping_lay_imp`

        Returns
        -------
        data : xr.Dataset
            Green's function, self-energy and occupation of the iteration.

        """
        # perform self-consistency loop
        data = self.update(layer_config=layer_config, n_process=n_process,
                           solve=self.default_solver if solver is None else solver,
                           **qmc_params)

        update_kdws = self.update.keywords  # pylint: disable=no-member
        siams = self.get_impurity_models(self_z=data.self_iw, gf_z=data.gf_iw, occ=data.occ)
        update_kdws.update(siams=siams, self_iw=data.self_iw, occ=data.occ)
        update_kdws['it'] += 1
        return data

    def mixed_iteration(self, mixing, *args, **kwds):
        """Call `Runner.iteration` and mix the hybridization functions **afterwards**."""
        update_kdws = self.update.keywords  # pylint: disable=no-member
        siams, old_siams = tee(update_kdws['siams'])
        update_kdws['siams'] = siams
        data = self.iteration(*args, **kwds)
        new_siams = update_kdws['siams']
        update_kdws['siams'] = mixed_siams(mixing, new=new_siams, old=old_siams)
        return data


# To register with at_exit to inform once that calculation is over
_finished_message = chain((lambda: LOGGER.progress('Finished calculation'),), repeat(lambda: None))
