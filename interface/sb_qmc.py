"""Utilities to interact with Junya's **spinboson** code for R-DMFT."""
import textwrap

from pathlib import Path

import numpy as np

import gftools as gt

from ..model import Hubbard_Parameters, sigma

N_IW = 1024


OUTPUT_DIR = "output"
INIT_FILE = "sb_qmc_param.init"

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
     1.00
     0.0
     0.0
     1  1.0  1.0
    #----------------------------------------------------------
    # N_BIN  N_MSR
    # N_SEG  N_SPIN  N_BOSON  N_SHIFT  N_SEGSP (optimize if <0)
     10  100000
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


def get_path(dir_) -> Path:
    """Return a Path object, asserting that the path exists."""
    dir_ = Path(dir_).expanduser()
    if not dir_.exists():
        raise OSError(f"Not a valid directory: {dir_}")
    return dir_


def setup(prm: Hubbard_Parameters, layer: int, gf_iw, self_iw, dir_='.'):
    """Prepare the 'hybrid_fct' file and parameters to use spinboson code."""
    assert gf_iw.ndim == 2, f"Dimension must be 2: (N_spins, N_iw), ndim: {gf_iw.ndim}"
    assert gf_iw.shape[0] == 2, f"First dimension must be of length 2=#Spins, "\
        f"here: {gf_iw.shape[0]}"
    assert gf_iw.shape == self_iw.shape, "Shape of self-energy and Gf have to match, " \
        f"Gf: {gf_iw.shape}, self: {self_iw.shape}"
    dir_ = get_path(dir_)
    # trim Green's function to spinboson code
    gf_iw, self_iw = gf_iw[:, :N_IW], self_iw[:, :N_IW]

    iw = gt.matsubara_frequencies(np.arange(gf_iw.shape[1]), beta=prm.beta)
    on_site_e = prm.onsite_energy()[:, layer]
    h_l = prm.h[layer]
    assert on_site_e.up - sigma.up*h_l == on_site_e.dn - sigma.dn*h_l
    hybrid_iw = iw + on_site_e[:, np.newaxis] - self_iw - 1./gf_iw
    # -2.07144315891833e-01  << 21
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
    init_content = PARAM_TEMPLATE.format(D=prm.D, T=prm.T, U=prm.U[layer],
                                         ef=-on_site_e.up + sigma.up*h_l,
                                         h=h_l)
    (dir_ / INIT_FILE).write_text(init_content)
