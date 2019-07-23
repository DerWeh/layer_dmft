"""Utilities to interact with **w2dynamics** CT-Hyb solver."""
import logging

from datetime import datetime
from functools import partial
from typing import Any, Dict
from pathlib import Path
from collections import ChainMap
from collections.abc import Mapping

import configobj
import numpy as np
import h5py

from layer_dmft import __version__, SIAM, dataio, fft, high_frequency_moments as hfm
from layer_dmft.util import Spins

LOGGER = logging.getLogger(__name__)

W2DYN_EXECUTABLE = '/home/andreasw/.pyenv/shims/python ' \
    + str(Path('~/workspace/code/w2dynamics_p3/cthyb').expanduser())
PARAMETERFILE = 'Parameters.in'
OUTPUT_FILE = "w2d_output.txt"

CFG = configobj.ConfigObj()

CFG.filename = PARAMETERFILE
CFG['General'] = {
    'DOS': 'readDelta',
    'DMFTsteps': 0,
    'StatisticSteps': 1,
    'NAt': 1,
    'muimpFile': 'mu_imp',
    'magnetism': 'ferro',  # magnetism will be handled by layer_dmft itself
    'FileNamePrefix': 'w2d-data',
}
CFG['Atoms'] = {}
CFG['Atoms']['1'] = {
    'Nd': 1,  # single Band
    'Hamiltonian': 'Density',
}
CFG['QMC'] = {
    'MeasGiw': 1,
    # 'WormMeasGiw': 1,
    # 'MeasGSigmaiw': 1,  # causes issues
    # 'WormMeasGSigmaiw': 1,
}

CFG_WORM = configobj.ConfigObj()

CFG_WORM.filename = PARAMETERFILE
CFG_WORM['General'] = CFG['General'].copy()
CFG_WORM['General']['StatisticSteps'] = 0
CFG_WORM['General']['WormSteps'] = 1
CFG_WORM['General']['FileNamePrefix'] = 'w2d-data-worm'
CFG_WORM['Atoms'] = {'1': CFG['Atoms']['1'].copy()}
CFG_WORM['QMC'] = {
    'WormMeasGiw': 1,
    'WormMeasGtau': 1,
    'WormMeasGSigmaiw': 1,
    'PercentageWormInsert': 0.3,
    'PercentageWormReplace': 0.1,
    'WormComponents': (1, 4),
    'WormSearchEta': 1,
}


class QMCParams(Mapping):
    """Non-physical parameters passed to **w2dynamics** CT-Hyb.

    See `QMCParams.__init__` for the meaning of the arguments.
    """

    __slots__ = ('Nwarmups', 'Nmeas', 'NCorr', 'Ntau', 'Niw',
                 'NLegMax', 'NLegOrder', 'MeasDensityMatrix', 'Eigenbasis',)
    __getitem__ = object.__getattribute__
    __setitem__ = object.__setattr__

    def __init__(self, Nwarmups: int, Nmeas: int, NCorr: int,
                 Ntau: int, Niw: int,
                 MeasDensityMatrix: int = 1, Eigenbasis: int = 1) -> None:
        """Initialize CT-Hyb parameters defining what will be sampled.

        Parameters
        ----------
        Nwarmups : int
            Number of of warm-up steps.
        Nmeas : int
            Number of measurements.
        NCorr : int
            Number of updates between measurements.
        Ntau : int
            Number of τ-points.
        Niw : int
            Number of (positive) Matsubara frequencies.
        MeasDensityMatrix : int
            Measure the density matrix :math:`⟨c^†_i c_j⟩`
        Eigenbasis : int
            XXX

        """
        self.Nwarmups = Nwarmups
        self.Nmeas = Nmeas
        self.NCorr = NCorr
        self.Ntau = Ntau
        self.Niw = Niw
        self.NLegMax = 1
        self.NLegOrder = 1
        self.MeasDensityMatrix = MeasDensityMatrix
        self.Eigenbasis = Eigenbasis

    def __len__(self):
        """Return the number of attributes in `__slots__`."""
        return len(self.__slots__)

    def __iter__(self):
        """Iterate over attributes in `__slots__`."""
        return iter(self.__slots__)

    @classmethod
    def slots(cls) -> set:
        """Return the set of existing attributes."""
        return set(cls.__slots__)


DEFAULT_QMC_PARAMS = QMCParams(
    Nwarmups=10**5, Nmeas=10**5, NCorr=50,
    Ntau=2049, Niw=1024,
)


def write_hybridization_iw(iws, hybrid_iw):
    r"""Write 'deltaiw' file containing hybridization function of the SIAM.

    Parameters
    ----------
    iws : (N_iw) complex np.ndarray
        Fermionic Matsubara frequencies.
    hybrid_iw : (2, N_iw) complex np.ndarray
        Hybridization function :math:`Δ(iω_n)` evaluated at Matsubara
        frequencies. It is necessary to have `N_iw >= N_IW`.

    """
    N_IW = DEFAULT_QMC_PARAMS.Niw
    # w2dynamics expects conjugated quantity
    hybrid_iw = hybrid_iw[:, :N_IW].conj()
    assert hybrid_iw.ndim == 2, f"Dimension must be 2: (N_spins, N_iw), ndim: {hybrid_iw.ndim}"
    digits = 14
    fill = 2 + 2 + digits + 4
    header = (
        'w_n'.ljust(fill)
        + 'Re spin up'.ljust(fill)
        + 'Im spin up'.ljust(fill)
        + 'Re spin dn'.ljust(fill)
        + 'Im spin dn'.ljust(fill)
    )
    np.savetxt("deltaiw", np.array([
        iws.imag,
        hybrid_iw[Spins.up].real, hybrid_iw[Spins.up].imag,
        hybrid_iw[Spins.dn].real, hybrid_iw[Spins.dn].imag,
    ]).T, fmt=f'%+.{digits}e', delimiter=' ', header=header)


def write_hybridization_tau(tau, hybrid_tau):
    r"""Write 'hybrid_tau' file containing hybridization function of the SIAM.

    Parameters
    ----------
    tau : (N_tau) float np.ndarray
        Tau mesh.
    hybrid_tau : (2, N_tau) float np.ndarray
        Hybridization function Δ(τ) evaluated at τ points [0, β].
        It is necessary to have `N_tau = N_TAU`.

    """
    # w2dynamics expects negative quantity
    N_TAU = DEFAULT_QMC_PARAMS.Ntau
    hybrid_tau = -hybrid_tau[..., ::-1]
    assert hybrid_tau.shape[-1] == N_TAU
    digits = 14
    fill = 2 + digits + 4
    header = (
        'tau'.ljust(fill)
        + 'spin up'.ljust(fill)
        + 'spin dn'.ljust(fill)
    )
    np.savetxt("deltatau", np.array([tau, hybrid_tau[Spins.up], hybrid_tau[Spins.dn]]).T,
               fmt=f'%+.{digits}e', delimiter=' ', header=header)


def get_path(dir_) -> Path:
    """Return a Path object, asserting that the path exists."""
    dir_ = Path(dir_).expanduser()
    if not dir_.exists():
        raise OSError(f"Not a valid directory: {dir_}")
    return dir_


def setup(siam: SIAM, dir_='.', worm=False, **kwds):
    """Prepare the input files to use **w2dynamics** code.

    Parameters
    ----------
    siam: SIAM
        The effective single impurity Anderson model to solve.
    dir_ :
        The working directory for the **spinboson** solver.
    kwds :
        Additional parameters for the CT-QMC. See `QMCParams`

    """
    if set(kwds.keys()) - QMCParams.slots():
        raise TypeError(f"Unknown keyword arguments: {kwds.keys()-QMCParams.slots()}")
    dir_ = get_path(dir_)

    hybrid_tau = siam.hybrid_tau()
    tau = np.linspace(0, siam.beta, num=hybrid_tau.shape[-1], endpoint=True)
    write_hybridization_tau(tau, hybrid_tau)
    write_hybridization_iw(siam.z, siam.hybrid_fct)
    e_onsite = siam.e_onsite
    cfg = CFG_WORM if worm else CFG
    np.savetxt(CFG['General']['muimpFile'], [e_onsite[Spins.up], e_onsite[Spins.dn]])
    # THIS OVERWRITES THE CONFIG!
    cfg['QMC'].update(kwds)
    cfg['General']['beta'] = siam.beta
    cfg['Atoms']['1']['Udd'] = siam.U
    cfg.write()


def run(dir_=".", n_process=1):
    """Execute the **spinboson** code."""
    from subprocess import CalledProcessError, Popen

    dir_ = get_path(dir_)
    command = f"mpirun -n {n_process} {W2DYN_EXECUTABLE}"
    with open(OUTPUT_FILE, "w") as outfile:
        proc = Popen(command.split(), stdout=outfile, stderr=outfile)
        try:
            proc.wait()
        except:
            proc.kill()
            proc.wait()
            raise
        retcode = proc.poll()
        if retcode:
            raise CalledProcessError(retcode, proc.args)


def get_last_output(dir_='.', cfg=CFG):
    """Return filename of last hdf5 output solely determined by the date."""
    dir_ = Path(dir_).absolute()
    pattern = cfg['General']['FileNamePrefix']+'*.hdf5'
    # FIXME: very brittle, depends on naming
    last = next(ofile for ofile in sorted(dir_.glob(pattern), reverse=True)
                if ofile.name.find('_lay'))
    return last


def check_consistency(siam: SIAM, h5output, N_iw: int):
    consistent = True
    if not np.allclose(siam.hybrid_fct.conj(), h5output['fiw/value'][0, :, N_iw:]):
        LOGGER.warning('Δ(iω_n) of w2dynamics solver and layer_dmft mismatch.')
        consistent = False
    if not np.allclose(siam.hybrid_tau(), -h5output['ftau/value'][0][..., ::-1]):
        LOGGER.warning('Δ(τ) of w2dynamics solver and layer_dmft mismatch.')
        consistent = False
    if not np.allclose(siam.gf0(), h5output['g0iw/value'][0, :, N_iw:]):
        LOGGER.warning('G0_imp pf w2dynamics solver and layer_dmft mismatch.')
        consistent = False
    return consistent


def save_data(siam: SIAM, dir_='.', name='w2d', compress=True, qmc_params=DEFAULT_QMC_PARAMS
              ) -> Dict[str, Any]:
    """Read the **spinboson** data and save it as numpy arrays."""
    data: Dict[str, Any] = {}
    data['solver'] = __name__
    data['__version__'] = __version__
    data['__date__'] = datetime.now().isoformat()
    N_iw = qmc_params['Niw']

    with h5py.File(get_last_output(dir_), mode='r') as h5file:
        output = h5file['stat-last/ineq-001/']
        # only positive frequencies
        data['tau'] = np.linspace(0, siam.beta, num=N_iw, endpoint=True)
        data['gf_tau'], data['gf_tau_err'] = -output['gtau/value'][0], output['gtau/error'][0]
        assert np.all(-data['gf_tau'][..., -1] - data['gf_tau'][..., 0] > 0), "Should be 1"

        # data['gf_x_self_tau'] = XXX
        data['hybrid_iw'] = siam.hybrid_fct
        assert np.all(data['hybrid_iw'][..., 0].imag < 0), "causality -> negative imaginary part"

        data['occ'] = np.diagonal(output['occ/value'][0, :, 0, :])
        data['occ_err'] = np.diagonal(output['occ/error'][0, :, 0, :])
        self_m0 = hfm.self_m0(siam.U, data['occ'][::-1])
        self_m1 = hfm.self_m1(siam.U, data['occ'][::-1])
        gf_m2 = -siam.e_onsite + self_m0
        gf_x_self_m1 = hfm.gf_x_self_m1(self_m0)
        gf_x_self_m2 = hfm.gf_x_self_m2(self_m0, self_m1, gf_m2)

        dft = partial(fft.dft_tau2iw, beta=siam.beta)
        data['gf_iw'] = dft(data['gf_tau'], moments=[(1, 1), gf_m2])
        assert np.all(data['gf_iw'][..., 0].imag < 0), "causality -> negative imaginary part"
        # gf_x_self_iw = dft(data['gf_x_self_tau'], moments=[gf_x_self_m1, gf_x_self_m2]) XXX

        # data['gf_x_self_iw'] = XXX
        # data['self_energy_iw'] = XXX

        data['self_energy_iw'] = output['siw/value'][0, :, N_iw:]  # FIXME
        assert np.all(data['self_energy_iw'][..., 0].imag < 0) \
            , "causality -> negative imaginary part"

        data['gf_iw_solver'] = output['giw/value'][0, :, N_iw:]  # just for debugging
        data['self_energy_iw_solver'] = output['siw/value'][0, :, N_iw:]  # just for debugging
        data['qmc_params'] = dict(qmc_params)
        if not check_consistency(siam, output, N_iw=N_iw):
            raise RuntimeError("Interface seems to be broken.")
        dataio.save_data(dir_=Path(dir_).expanduser()/dataio.IMP_OUTPUT, name=name,
                         compress=compress, **data)
    return data


def get_worm(container, mask=()):
    components = ('00001', '00004')
    val = np.stack([container[f"{component}/value"][mask] for component
                    in components])
    err = np.stack([container[f"{component}/error"][mask] for component
                    in components])
    return val, err


def save_worm_data(siam: SIAM, dir_='.', name='w2d', compress=True,
                   qmc_params=DEFAULT_QMC_PARAMS) -> Dict[str, Any]:
    """Read the **spinboson** data and save it as numpy arrays."""
    data: Dict[str, Any] = {}
    data['solver'] = __name__
    data['__version__'] = __version__
    data['__date__'] = datetime.now().isoformat()
    N_iw = qmc_params['Niw']
    N_tau = qmc_params['Ntau']

    # TODO: write unit test for non-interacting case
    with h5py.File(get_last_output(dir_, CFG_WORM), mode='r') as h5file:
        output = h5file['worm-last/ineq-001/']
        # only positive frequencies
        data['tau'] = np.linspace(0, siam.beta, num=N_tau, endpoint=True)
        data['gf_tau'], data['gf_tau_err'] = get_worm(output['gtau-worm'])
        data['gf_tau'] *= -1
        assert np.all(-data['gf_tau'][..., -1] - data['gf_tau'][..., 0] > 0), "Should be 1"

        data['hybrid_iw'] = siam.hybrid_fct
        assert np.all(data['hybrid_iw'][..., 0].imag < 0), "causality -> negative imaginary part"

        data['occ'] = -data['gf_tau'][:, -1]
        data['occ_err'] = -data['gf_tau_err'][:, -1]

        data['gf_iw'], data['gf_iw_err'] = get_worm(output['giw-worm'], mask=slice(N_iw, None))
        assert np.all(data['gf_iw'][..., 0].imag < 0), "causality -> negative imaginary part"
        data['gf_x_self_iw'], data['gf_x_self_iw_err'] = get_worm(output['gsigmaiw-worm'], mask=slice(N_iw, None))
        # gf_x_self_iw = dft(data['gf_x_self_tau'], moments=[gf_x_self_m1, gf_x_self_m2]) XXX


        data['self_energy_iw'] = -data['gf_x_self_iw']/data['gf_iw']
        assert np.all(data['self_energy_iw'][..., 0].imag < 0) \
            , "causality -> negative imaginary part"

        data['qmc_params'] = dict(qmc_params)
        if not check_consistency(siam, output, N_iw=N_iw):
            raise RuntimeError("Interface seems to be broken.")
        dataio.save_data(dir_=Path(dir_).expanduser()/dataio.IMP_OUTPUT, name=name,
                         compress=compress, **data)
    return data


def solve(siam: SIAM, n_process, output_name, dir_='.', worm=False, **kwds):
    if set(kwds.keys()) - QMCParams.slots():
        raise TypeError(f"Unknown keyword arguments: {kwds.keys()-QMCParams.slots()}")
    solver_kwds = ChainMap(kwds, DEFAULT_QMC_PARAMS)
    setup(siam, dir_=dir_, worm=worm, **solver_kwds)
    run(n_process=n_process, dir_=dir_)
    _save_data = save_worm_data if worm else save_data
    data = _save_data(siam, name=output_name, dir_=dir_, qmc_params=solver_kwds)
    return data
