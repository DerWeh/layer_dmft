"""Utilities to interact with Junya's **spinboson** code for R-DMFT."""
import textwrap
from typing import Dict, Any
from functools import partial

from pathlib import Path
from datetime import date, datetime
from collections import ChainMap, Mapping, namedtuple

import numpy as np

import gftools as gt

from layer_dmft import __version__, fft, high_frequency_moments as hfm, dataio
from layer_dmft.util import SpinResolvedArray
from layer_dmft.model import SIAM, SIGMA

N_TAU = 2048
N_IW = 1024  # TODO: scan code for proper number

SB_EXECUTABLE = Path('~/spinboson-1.10/sb_qmc.out').expanduser()
OUTPUT_DIR = "output"
OUTPUT_FILE = "output.txt"
INIT_FILE = "sb_qmc_param.init"
GF_IW_FILE = "00-Gf_omega.dat"
GF_TAU_FILE = "00-Gf_tau.dat"
SELF_FILE = "00-self.dat"
SUSCEPT_IW_FILE = "00-chi_omega.dat"
SUSCEPT_TAU_FILE = "00-chi_omega.dat"

IM_STEP = 2

PARAM_TEMPLATE = textwrap.dedent(
    r"""    #
    #
    #----------------------------------------------------------
    # i_program
    # flag_tp
     0
     {FLAG_TP}
    #----------------------------------------------------------
    # D
    # T
    # ef  h (= ef_d - ef_u)
    # U
     1.
     {T}
     {ef}  {h}
     {U}
    #----------------------------------------------------------
    # V^2
    # g_ch^2
    # g_xy^2  [g_z^2]  (default: g_z = g_xy)
    # b_dos  [w0 (if b_dos=0)]  [w_cutoff power (if b_dos=1)]
     {V2_up}  {V2_dn}
     0.0
     0.0
     1  1.0  1.0
    #----------------------------------------------------------
    # N_BIN  N_MSR
    # N_SEG  N_SPIN  N_BOSON  N_SHIFT  N_SEGSP (optimize if <0)
     {N_BIN}  {N_MSR}
     -3  -3  -10  -6  -10
    #----------------------------------------------------------
    # i_mesh  N_X  X_MIN  X_MAX  (if i_program != 0)
     1  11  0.01  1
    #----------------------------------------------------------



    #==========================================================
    #
    # i_program
    # // 0: "Parameters fixed",
    # // 1: "T varies",
    # // 2: "V^2 varies",
    # // 3: "g^2 (= g_xy = g_z) varies",
    # // 4: "g_xy^2 varies",
    # // 5: "g_z^2 varies",
    # // 6: "ef varies",
    # // 7: "h varies",
    # // 8: "w0 varies",
    # // 9: "gamma varies",
    # flag_tp
    # // 0: "Green function",
    # // 1: "Green function, susceptibility",
    # // 2: "Green function, susceptibility, vertex",
    # // 3: "free-energy by Wang-Landau sampling",
    #
    # b_dos
    # // 0: DOS = delta(w-w0);
    # // 1: DOS \propto w^{{gamma}}  (with cutoff `w_cutoff')
    #
    # i_mesh
    # // 0: linear mesh
    # // 1: logarithmic mesh
    #"""
)


class QMCParams(Mapping):
    """Non-physical parameters passed to **spinboson** CT-Hyb.

    See `QMCParams.__init__` for the meaning of the arguments.
    """

    __slots__ = ('N_BIN', 'N_MSR', 'FLAG_TP')
    __getitem__ = object.__getattribute__
    __setitem__ = object.__setattr__

    def __init__(self, N_BIN: int, N_MSR: int, FLAG_TP: int = 0) -> None:
        """Initialize CT-Hyb parameters defining what will be sampled.

        Parameters
        ----------
        N_BIN : int
            Number of bins used in CT-Hyb.
        N_MSR : int
            Total number of measurements performed.
        FLAG_TP : int
            If two particle quantities will be sampled.
            0: Green's function
            1: Green's function & susceptibility
            2: Green's function & susceptibility & vertex
            3: free energy by Wang-Landau sampling
            For DMFT only the Green's function is necessary, the self-energy
            will also be sampled.

        """
        self.N_BIN = int(N_BIN)
        self.N_MSR = int(N_MSR)
        if not isinstance(FLAG_TP, int) or FLAG_TP < 0:
            raise TypeError(f"'FLAG_TP' must be non-negative integer, got: {FLAG_TP}")
        self.FLAG_TP = FLAG_TP

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


DEFAULT_QMC_PARAMS = QMCParams(N_BIN=10, N_MSR=10**5)


def get_path(dir_) -> Path:
    """Return a Path object, asserting that the path exists."""
    dir_ = Path(dir_).expanduser()
    if not dir_.exists():
        raise OSError(f"Not a valid directory: {dir_}")
    return dir_


def write_hybridization_iw(hybrid_iw):
    r"""Write 'hybrid_fct' file containing hybridization function of the SIAM.

    Parameters
    ----------
    hybrid_iw : (2, N_iw) complex np.ndarray
        Hybridization function :math:`\Delta(i\omega_n)` evaluated at matsubara
        frequencies. It is necessary to have `N_iw >= N_IW`.

    """
    hybrid_iw = hybrid_iw[:, :N_IW]
    assert hybrid_iw.ndim == 2, f"Dimension must be 2: (N_spins, N_iw), ndim: {hybrid_iw.ndim}"
    assert hybrid_iw.shape[0] == 2, ("First dimension must be of length 2=#Spins, "
                                     f"here: {hybrid_iw.shape[0]}")
    digits = 14
    fill = 2 + 2 + digits + 4
    header = (
        'Re spin up'.ljust(fill)
        + 'Im spin up'.ljust(fill)
        + 'Re spin dn'.ljust(fill)
        + 'Im spin dn'.ljust(fill)
    )
    np.savetxt("hybrid_fct.dat", hybrid_iw.T,
               fmt=[f'%+.{digits}e %+.{digits}e', ]*2, delimiter=' ',
               header=header)


def write_hybridization_tau(hybrid_tau):
    r"""Write 'hybrid_tau' file containing hybridization function of the SIAM.

    Parameters
    ----------
    hybrid_tau : (2, N_tau) float np.ndarray
        Hybridization function Δ(τ) evaluated at τ points [0, β].
        It is necessary to have `N_tau = N_TAU + 1`.

    """
    assert hybrid_tau.shape == (2, N_TAU + 1)
    digits = 14
    fill = 2 + digits + 4
    header = (
        'spin up'.ljust(fill)
        + 'spin dn'.ljust(fill)
    )
    np.savetxt("hybrid_tau.dat", hybrid_tau.T,
               fmt=[f'%+.{digits}e', ]*2, delimiter=' ',
               header=header)


def setup(siam: SIAM, dir_='.', **kwds):
    """Prepare the 'hybrid_fct' file and parameters to use **spinboson** code.

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

    on_site_e = siam.e_onsite
    h_l = on_site_e.up - on_site_e.dn
    assert np.allclose(on_site_e.up - SIGMA.up*h_l, on_site_e.dn - SIGMA.dn*h_l,
                       rtol=1e-12, atol=1e-15)

    write_hybridization_tau(siam.hybrid_tau())
    (dir_ / OUTPUT_DIR).mkdir(exist_ok=True)
    init_content = PARAM_TEMPLATE.format(T=siam.T, U=siam.U,
                                         ef=-on_site_e.up + SIGMA.up*h_l, h=h_l,
                                         V2_up=1, V2_dn=1,  # not in use
                                         **kwds)
    (dir_ / INIT_FILE).write_text(init_content)


def run(dir_=".", n_process=1):
    """Execute the **spinboson** code."""
    from subprocess import Popen

    dir_ = get_path(dir_)
    command = f"mpirun -n {n_process} {SB_EXECUTABLE}"
    with open(OUTPUT_FILE, "w") as outfile:
        proc = Popen(command.split(), stdout=outfile)
        try:
            proc.wait()
        except Exception as exc:
            proc.terminate()
            raise exc


def solve(siam: SIAM, n_process, output_name, dir_='.', **kwds):
    if set(kwds.keys()) - QMCParams.slots():
        raise TypeError(f"Unknown keyword arguments: {kwds.keys()-QMCParams.slots()}")
    solver_kwds = ChainMap(kwds, DEFAULT_QMC_PARAMS)
    setup(siam, dir_=dir_, **solver_kwds)
    run(n_process=n_process, dir_=dir_)
    data = save_data(siam, name=output_name, dir_=dir_, qmc_params=solver_kwds)
    return data


def output_dir(dir_) -> Path:
    """Return the output directory of the **spinboson** code.

    Parameters
    ----------
    dir_ : str or Path
        Output directory is calculated relative to `dir_`

    Returns
    -------
    output_dir : Path
        The output directory.

    Raises
    ------
    ValueError
        If no output directory can be found in `dir_`.

    """
    dir_path = Path(dir_).expanduser()
    if dir_path.name == OUTPUT_DIR:  # already in output directory
        return dir_path
    # descend into output directory
    dir_path /= OUTPUT_DIR
    if not dir_path.is_dir():
        raise ValueError("Non output directory can be found in " + str(dir_))
    return dir_path


def read_tau(dir_='.') -> np.ndarray:
    """Return the imaginary time mesh from file in `dir_`.

    Parameters
    ----------
    dir_ : str or Path
        The directory where the output of the **spinboson** code is located.

    Returns
    -------
    tau : (N_tau) np.ndarray
        The imaginary time mesh.

    """
    out_dir = output_dir(dir_)
    tau = np.loadtxt(out_dir / GF_TAU_FILE, unpack=True, usecols=0)
    assert tau.ndim == 1
    return tau


def read_gf_tau(dir_='.') -> gt.Result:
    """Return the imaginary time Green's function from file in `dir_`.

    Parameters
    ----------
    dir_ : str or Path
        The directory where the output of the **spinboson** code is located.

    Returns
    -------
    gf_tau.x, gf_tau_err : (2, N_tau) util.SpinResolvedArray
        The imaginary time Green's function and its error.
        The shape of the arrays is (#spins, # imaginary time points).

    """
    out_dir = output_dir(dir_)
    gf_output = np.loadtxt(out_dir / GF_TAU_FILE, unpack=True, usecols=range(1, 5))
    assert gf_output.shape[0] == 4
    gf_tau = gf_output[::2].view(type=SpinResolvedArray)
    gf_tau_err = gf_output[1::2].view(type=SpinResolvedArray)
    return gt.Result(x=gf_tau, err=gf_tau_err)


def read_gf_x_self_tau(dir_='.') -> SpinResolvedArray:
    """Return the convolution (Gf x self energy)(tau) from file in `dir_`.

    Parameters
    ----------
    dir_ : str or Path
        The directory where the output of the **spinboson** code is located.

    Returns
    -------
    gf_x_self_tau : (2, N_tau) util.SpinResolvedArray
        The convolution of Green's function and self energy as a function of
        imaginary time tau.
        The shape of the arrays is (#spins, # imaginary time points).

    """
    out_dir = output_dir(dir_)
    gf_output = np.loadtxt(out_dir / GF_TAU_FILE, unpack=True, usecols=(5, 6))
    assert gf_output.shape[0] == 2
    gf_x_self_tau = gf_output.view(type=SpinResolvedArray)
    return gf_x_self_tau


def read_gf_iw(dir_='.') -> SpinResolvedArray:
    """Return the Matsubara Green's function from file in `dir_`.

    Parameters
    ----------
    dir_ : str or Path
        The directory where the output of the **spinboson** code is located.

    Returns
    -------
    gf_iw : (2, N_iw) util.SpinResolvedArray
        The Matsubara Green's function. The shape of the array is
        (#spins, #Matsubara frequencies).

    """
    out_dir = output_dir(dir_)
    gf_output = np.loadtxt(out_dir / GF_IW_FILE, unpack=True)
    gf_iw_real = gf_output[1::IM_STEP]
    gf_iw_imag = gf_output[2::IM_STEP]
    gf_iw = gf_iw_real + 1j*gf_iw_imag
    assert gf_iw.shape[0] == 2
    gf_iw = gf_iw.view(type=SpinResolvedArray)
    return gf_iw


def read_self_energy_iw(dir_='.') -> SpinResolvedArray:
    """Return the self-energy from file in `dir_`.

    Parameters
    ----------
    dir_ : str or Path
        The directory where the output of the **spinboson** code is located.

    Returns
    -------
    self_iw : (2, N_iw) util.SpinResolvedArray
        The self-energy. The shape of the array is
        (#spins, #Matsubara frequencies).

    """
    out_dir = output_dir(dir_)
    gf_output = np.loadtxt(out_dir / SELF_FILE, unpack=True, usecols=range(5, 9))
    self_real = gf_output[0::IM_STEP]
    self_imag = gf_output[1::IM_STEP]
    self_iw = self_real + 1j*self_imag
    assert self_iw.shape[0] == 2
    self_iw = self_iw.view(type=SpinResolvedArray)
    return self_iw


Suscept = namedtuple('Susceptibility', ['spin', 'charge'])


def read_susceptibility_tau(dir_='.') -> Suscept:
    """Return the imaginary time susceptibility from file in `dir_`.

    Parameters
    ----------
    dir_ : str or Path
        The directory where the output of the **spinboson** code is located.

    Returns
    -------
    susceptibility.spin, susceptibility.charge: gt.Result
        The spin- (charge-) susceptibility in τ-space and its error.
        The value (`gt.Result.x`) and the error (`gt.Result.err`) are
        1d float `np.ndarray`s.

    """
    out_dir = output_dir(dir_)
    suscept_output = np.loadtxt(out_dir / SUSCEPT_TAU_FILE, unpack=True, usecols=range(1, 5))
    spin = gt.Result(x=suscept_output[0], err=suscept_output[1])
    charge = gt.Result(x=suscept_output[2], err=suscept_output[3])
    return Suscept(spin=spin, charge=charge)


def read_susceptibility_iw(dir_='.') -> Suscept:
    """Return the susceptibility from file in `dir_` for *bosonic* frequencies.

    Parameters
    ----------
    dir_ : str or Path
        The directory where the output of the **spinboson** code is located.

    Returns
    -------
    susceptibility.spin, susceptibility.charge : (N_iw, ) float np.ndarray
        The spin- (charge-) susceptibility for *bosonic* Matsubara frequencies.

    """
    out_dir = output_dir(dir_)
    suscept_output = np.loadtxt(out_dir / SUSCEPT_IW_FILE, unpack=True, usecols=[1, 3])  # FIXME
    return Suscept(spin=suscept_output[0], charge=suscept_output[1])


def read_occ(dir_='.') -> gt.Result:
    """Return the occupation number from file in `dir_`.

    Parameters
    ----------
    dir_ : str or Path
        The directory where the output of the **spinboson** code is located.

    Returns
    -------
    occ.x, occ.err : (2, ) util.SpinResolvedArray
        The occupation number and its error.

    """
    gf_tau = read_gf_tau(dir_)
    # occ = Gf(τ=0^-) = Gf(τ=β^-)
    return gt.Result(x=-gf_tau.x[:, -1], err=gf_tau.err[:, -1])


def save_data(siam: SIAM, dir_='.', name='sb', compress=True, qmc_params=DEFAULT_QMC_PARAMS):
    """Read the **spinboson** data and save it as numpy arrays."""
    data: Dict[str, Any] = {}
    data['solver'] = __name__
    data['__version__'] = __version__
    data['__date__'] = datetime.now().isoformat()
    data['tau'] = read_tau(dir_)
    data['gf_tau'], data['gf_tau_err'] = read_gf_tau(dir_)
    data['gf_x_self_tau'] = read_gf_x_self_tau(dir_)

    occ_other = -data['gf_tau'][::-1, -1]
    self_m0 = hfm.self_m0(siam.U, occ_other)
    self_m1 = hfm.self_m1(siam.U, occ_other)
    gf_m2 = -siam.e_onsite + self_m0
    gf_x_self_m1 = hfm.gf_x_self_m1(self_m0)
    gf_x_self_m2 = hfm.gf_x_self_m2(self_m0, self_m1, gf_m2)

    dft = partial(fft.dft_tau2iw, beta=siam.beta)
    gf_iw = dft(data['gf_tau'], moments=[(1, 1), gf_m2])
    gf_x_self_iw = dft(data['gf_x_self_tau'], moments=[gf_x_self_m1, gf_x_self_m2])

    data['gf_iw'] = gf_iw
    data['gf_x_self_iw'] = gf_x_self_iw
    data['self_energy_iw'] = gf_x_self_iw/gf_iw

    data['gf_iw_solver'] = read_gf_iw(dir_)  # just for debugging
    data['self_energy_iw_solver'] = read_self_energy_iw(dir_)  # just for debugging
    data['misc'] = np.genfromtxt(output_dir(dir_)/'xx.dat', missing_values='?')
    data['qmc_params'] = dict(qmc_params)
    if 0 < qmc_params['FLAG_TP'] < 3:
        suscept_iw = read_susceptibility_iw()
        data['spin_susceptibility_iw'] = suscept_iw.spin
        data['charge_susceptibility_iw'] = suscept_iw.charge
        suscept_tau = read_susceptibility_tau()
        data['spin_susceptibility_tau'] = suscept_tau.spin.x
        data['spin_susceptibility_tau_err'] = suscept_tau.spin.err
        data['charge_susceptibility_tau'] = suscept_tau.charge.x
        data['charge_susceptibility_tau_err'] = suscept_tau.charge.err
    dataio.save_data(dir_=Path(dir_).expanduser()/dataio.IMP_OUTPUT, name=name, **data)
    return data
