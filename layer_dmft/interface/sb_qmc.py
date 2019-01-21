"""Utilities to interact with Junya's **spinboson** code for R-DMFT."""
import textwrap

from pathlib import Path
from datetime import date
from collections import Mapping

import numpy as np

import gftools as gt

from ..util import SpinResolvedArray
from ..model import Hubbard_Parameters, sigma

N_IW = 1024  # TODO: scan code for proper number


SB_EXECUTABLE = Path('~/spinboson-1.10/sb_qmc.out').expanduser()
OUTPUT_DIR = "output"
OUTPUT_FILE = "output.txt"
INIT_FILE = "sb_qmc_param.init"
GF_IW_FILE = "00-Gf_omega.dat"
GF_TAU_FILE = "00-Gf_tau.dat"
SELF_FILE = "00-self.dat"

IM_STEP = 2

PARAM_TEMPLATE = textwrap.dedent(
    r"""    #
    #
    #----------------------------------------------------------
    # i_program
    # flag_tp
     0
     0
    #----------------------------------------------------------
    # D
    # T
    # ef  h (= ef_d - ef_u)
    # U
     {D}
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


class _qmc_params(Mapping):
    __slots__ = ('N_BIN', 'N_MSR')
    __getitem__ = object.__getattribute__
    __setitem__ = object.__setattr__

    def __init__(self, N_BIN, N_MSR):
        self.N_BIN = N_BIN
        self.N_MSR = N_MSR

    def __len__(self):
        return len(self.__slots__)

    def __iter__(self):
        return iter(self.__slots__)

    def slots(self) -> set:
        """Return the set of existing attributes."""
        return set(self.__slots__)


QMC_PARAMS = _qmc_params(N_BIN=10, N_MSR=10**5)


def get_path(dir_) -> Path:
    """Return a Path object, asserting that the path exists."""
    dir_ = Path(dir_).expanduser()
    if not dir_.exists():
        raise OSError(f"Not a valid directory: {dir_}")
    return dir_


def setup(prm: Hubbard_Parameters, layer: int, gf_iw, self_iw, occ, dir_='.', **kwds):
    """Prepare the 'hybrid_fct' file and parameters to use **spinboson** code.

    Parameters
    ----------
    prm : Hubbard_Parameters
        The model parameters.
    layer : int
        The index of the layer that is mapped onto the impurity problem.
    gf_iw, self_iw : (2, N_IW) complex np.ndarray
        The local Matsubara Green's function and self energy of the layer.
    occ : (2, ) float np.ndarray
        The local occupation number.
    dir_ :
        The working directory for the **spinboson** solver.
    kwds :
        Additional parameters for the CT-QMC. See `QMC_PARAMS`

    """
    assert gf_iw.ndim == 2, f"Dimension must be 2: (N_spins, N_iw), ndim: {gf_iw.ndim}"
    assert gf_iw.shape[0] == 2, ("First dimension must be of length 2=#Spins, "
                                 f"here: {gf_iw.shape[0]}")
    assert gf_iw.shape == self_iw.shape, ("Shape of self-energy and Gf have to match, "
                                          f"Gf: {gf_iw.shape}, self: {self_iw.shape}")
    if set(kwds.keys()) - QMC_PARAMS.slots():
        raise TypeError("Unknown keyword arguments:"
                        f" {kwds.keys()-QMC_PARAMS.slots()}")
    dir_ = get_path(dir_)
    # trim Green's function to spinboson code
    gf_iw, self_iw = gf_iw[:, :N_IW], self_iw[:, :N_IW]

    iw = gt.matsubara_frequencies(np.arange(gf_iw.shape[1]), beta=prm.beta)
    on_site_e = prm.onsite_energy()[:, layer]
    h_l = prm.h[layer]
    assert on_site_e.up - sigma.up*h_l == on_site_e.dn - sigma.dn*h_l
    hybrid_iw = iw + on_site_e[:, np.newaxis] - self_iw - 1./gf_iw
    hybrid_m1 = prm.hybrid_fct_m1(occ)[:, layer]
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
    (dir_ / OUTPUT_DIR).mkdir(exist_ok=True)
    qmc_dict = dict(QMC_PARAMS)
    qmc_dict.update(kwds)
    init_content = PARAM_TEMPLATE.format(D=prm.D, T=prm.T, U=prm.U[layer],
                                         ef=-on_site_e.up + sigma.up*h_l, h=h_l,
                                         V2_up=hybrid_m1.up, V2_dn=hybrid_m1.dn,
                                         **qmc_dict)
    (dir_ / INIT_FILE).write_text(init_content)


def run(dir_=".", n_process=1):
    """Execute the **spinboson** code."""
    from subprocess import check_call
    dir_ = get_path(dir_)
    command = f"mpirun -n {n_process} {SB_EXECUTABLE}"
    with open(OUTPUT_FILE, "w") as outfile:
        check_call(command.split(), stdout=outfile)


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


def save_data(dir_='.', name='sb', compress=True):
    """Read the **spinboson** data and save it as numpy arrays."""
    dir_ = Path(dir_).expanduser()
    dir_.mkdir(exist_ok=True)
    data = {}
    data['tau'] = read_tau(dir_)
    data['gf_tau'], data['gf_tau_err'] = read_gf_tau(dir_)
    data['gf_x_self_tau'] = read_gf_x_self_tau(dir_)
    data['gf_iw'] = read_gf_iw(dir_)
    data['self_energy_iw'] = read_self_energy_iw(dir_)
    data['misc'] = np.genfromtxt(output_dir(dir_)/'xx.dat', missing_values='?')
    save_method = np.savez_compressed if compress else np.savez
    name = date.today().isoformat() + '_' + name
    (dir_/"imp_output").mkdir(exist_ok=True)
    save_method(dir_/"imp_output"/name, **data)
