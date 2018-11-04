#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : layer_hb.py
# Author            : Weh Andreas <andreas.weh@physik.uni-augsburg.de>
# Date              : 31.08.2018
# Last Modified Date: 31.10.2018
# Last Modified By  : Weh Andreas <andreas.weh@physik.uni-augsburg.de>
"""Utilities to interact with Junya's **layer_hb** code for R-DMFT.

This module contains function to read the output, and run the code.
"""
from pathlib import Path

import numpy as np

import gftools as gt

from .. import model
from ..util import SelfEnergy, SpinResolvedArray

DMFT_FILE = "00-dmft.dat"
GF_FILE = "00-G_omega.dat"
SELF_FILE = "00-self.dat"
OCC_FILE = "00-layer.dat"
OUTPUT_DIR = "output"
OUTPUT_FILE = "output.txt"
STEP = 8
IM_STEP = 2
# there are 8 different coulumns per layer and spin
# 1:\Re Δ(iω_n) 2:\Im Δ(iω_n) 3: 4: 5: 6:\Im F(iω_n) 7: 8:\Im G(iω_n)
DOS_DICT = {
    -1: model.hilbert_transform['chain'],
    1: model.hilbert_transform['bethe']
}


def output_dir(dir_) -> Path:
    """Return the output directory of the **layer_hb** code.

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


def find(fb, word):
    """Iterate stream `fb` until line starting with `word` is found and return it.

    `find` iterates through `fb` thus its position changes.

    Parameters
    ----------
    fb : iterable
        Stream (filebuffer) which is searched through. The method is intended
        to work with with objects generated by `open`.
    word : str
        The string the searched sentence starts with.

    Returns
    -------
    find : str
        The string starting with `word`.

    Raises
    ------
    ValueError
        If no string starting with `word` is found in `fb`.

    """
    for line in fb:
        if line.strip().startswith(word):
            break
    else:
        raise ValueError(f'{word} not in stream {fb}')
    return line  # pylint: disable=undefined-loop-variable


def get_scalar(line, dtype=np.float):
    """Return scalar of type `dtype` from `line` '... = scalar'."""
    return dtype(line.split('=')[1].strip())


def find_scaler(fb, word, dtype=np.float):
    """`find` `word` and `get_scalar`."""
    return get_scalar(find(fb, word), dtype=dtype)


def get_array(line, dtype=np.float):
    """Return array of type `dtype` from `line` '... = array'."""
    return np.fromstring(line.split('=')[1].strip(), sep=' ', dtype=dtype)


def find_array(fb, word, dtype=np.float):
    """`find` `word` and `get_array`."""
    return get_array(find(fb, word), dtype=dtype)


def load_param(dir_='.') -> model.Hubbard_Parameters:
    """Generate `model.Hubbard_Parameters` from `OUTPUT_FILE`.

    This reads the `OUTPUT_FILE` in `dir_`, or its subdirectory `dir_`/output
    and populates `model.Hubbard_Parameters` with the corresponding parameters.

    Parameters
    ----------
    dir_ : str or Path
        The directory where the output of the **layer_hb** code is located.

    Returns
    -------
    prm : model.Hubbard_Parameters
        The parameter object.

    """
    out_file = Path(dir_).expanduser() / OUTPUT_FILE
    assert out_file.is_file()
    prm = model.Hubbard_Parameters()
    with open(out_file, 'r') as out_fp:
        DOS = find_scaler(out_fp, 'DOS =', dtype=np.int)
        prm.hilbert_transform = DOS_DICT[DOS]

        find(out_fp, 'Layer configuration')
        imp_labels = find_array(out_fp, 'impurity label', dtype=np.int)
        N_l = imp_labels.size
        find(out_fp, 'transfer label')
        rhs = ' '.join(next(out_fp).strip() for __ in range(N_l))
        t_mat_labels = np.fromstring(rhs, sep=' ', dtype=np.int)
        t_mat_labels = t_mat_labels.reshape(N_l, N_l)
        find(out_fp, 'Parameters')
        prm.D = find_scaler(out_fp, 'D =', dtype=np.float)
        line = find(out_fp, 'T =')
        line = line.split('(')[0]  # remove parentheses (beta = ...)
        prm.T = get_scalar(line, dtype=np.float)
        t_values = find_array(out_fp, 't =', dtype=np.float)
        prm.t_mat = t_values[t_mat_labels]
        prm.U = find_array(out_fp, 'U =', dtype=np.float)[imp_labels]
        prm.mu = find_array(out_fp, 'mu =', dtype=np.float)[imp_labels]
        prm.h = find_array(out_fp, 'h =', dtype=np.float)[imp_labels]
        prm.V = np.zeros_like(prm.mu)
        prm.assert_valid()
    return prm


def read_imp_labels(dir_='.'):
    """Return the impurity labels from file in `dir_`.

    Parameters
    ----------
    dir_ : str or Path
        The directory where the output of the **layer_hb** code is located.

    Returns
    -------
    imp_labels : (N_imp, ) int ndarray
        The impurity labels relating layers to impurities.

    """
    out_file = Path(dir_).expanduser() / OUTPUT_FILE
    assert out_file.is_file()
    with open(out_file, 'r') as out_fp:
        find(out_fp, 'Layer configuration')
        imp_labels = find_array(out_fp, 'impurity label', dtype=np.int)
    return imp_labels


def expand_layers(axis, dir_):
    """Expand the `axis` corresponding to impurities to layers.

    Symmetry related layers will be mapped to the same impurity, reducing the
    size of the problem. `expand_layers` maps this back to the larger layer
    problem.

    Parameters
    ----------
    axis : int
        The axis which will be expanded.
    dir_ : str or Path
        The directory containing the information of the mapping `imp_labels`.

    Returns
    -------
    expand_layers : slice
        The slice which can be used on the data to expand it
        (data[`expand_layers`]).

    """
    imp_labels = read_imp_labels(dir_)
    return [slice(None, None), ]*axis + [imp_labels, ]


def reduce_layers(axis, dir_):
    """Reduces the `axis` corresponding to impurities to layers.

    Symmetry related layers will be mapped to the same impurity, reducing the
    size of the problem. This is the inverse of `expand_layers`.

    Parameters
    ----------
    axis : int
        The axis which will be expanded.
    dir_ : str or Path
        The directory containing the information of the mapping `imp_labels`.

    Returns
    -------
    reduce_layers : slice
        The slice which can be used on the data to reduce it
        (data[`reduce_layers`]).

    """
    imp_labels = read_imp_labels(dir_)
    __, indices = np.unique(imp_labels, return_index=True)
    return [slice(None, None), ]*axis + [indices, ]


def read_iw(dir_='.'):
    """Return the Matsubara frequencies."""
    prm = load_param(dir_)
    out_dir = output_dir(dir_)
    iw_output = np.loadtxt(out_dir / GF_FILE, unpack=True)
    return gt.matsubara_frequencies(iw_output[0], beta=prm.beta)


def read_gf_iw(dir_='.', expand=False) -> SpinResolvedArray:
    """Return the local Green's function from file in `dir_`.

    Parameters
    ----------
    dir_ : str or Path
        The directory where the output of the **layer_hb** code is located.
    expand : bool, optional
        Performs the symmetry expansion to return data for all layers.
        See `expand_layers`.

    Returns
    -------
    gf_iw : (2, N_l, N_iw) util.SpinResolvedArray
        The local Green's function. The shape of the array is
        (#spins, #layers, #Matsubara frequencies).

    """
    out_dir = output_dir(dir_)
    gf_output = np.loadtxt(out_dir / GF_FILE, unpack=True)
    gf_iw_real = gf_output[1::IM_STEP]
    gf_iw_imag = gf_output[2::IM_STEP]
    gf_iw = gf_iw_real + 1j*gf_iw_imag
    gf_iw = SpinResolvedArray(
        up=gf_iw[0::2],
        dn=gf_iw[1::2]
    )
    if expand:
        return gf_iw[expand_layers(axis=1, dir_=dir_)]
    return gf_iw


def read_self_energy_iw(dir_='.', expand=False) -> SelfEnergy:
    """Return the local self-energy from file in `dir_`.

    Parameters
    ----------
    dir_ : str or Path
        The directory where the output of the **layer_hb** code is located.
    expand : bool, optional
        Performs the symmetry expansion to return data for all layers.
        See `expand_layers`.

    Returns
    -------
    self_iw : (2, N_l, N_iw) util.SpinResolvedArray
        The self-energy. The shape of the array is
        (#spins, #layers, #Matsubara frequencies).

    """
    out_dir = output_dir(dir_)
    gf_output = np.loadtxt(out_dir / SELF_FILE, unpack=True)
    self_real = gf_output[1::IM_STEP]
    self_imag = gf_output[2::IM_STEP]
    self = self_real + 1j*self_imag
    self = SpinResolvedArray(
        up=self[0::2],
        dn=self[1::2]
    )
    prm = load_param(dir_)
    occ, __ = read_occ(dir_)
    U = prm.U
    if expand:
        self = self[expand_layers(axis=1, dir_=dir_)]
    else:
        U = U[reduce_layers(axis=0, dir_=dir_)]
        occ = occ[reduce_layers(axis=1, dir_=dir_)]
    self += .5 * U[:, np.newaxis]
    #     self_static = occ[::-1] * prm.U
    # else:
    #     self_static = (occ[::-1] * prm.U)[reduce_layers(axis=1, dir_=dir_)]
    # self += self_static[..., np.newaxis]
    return SelfEnergy(self, occupation=occ, interaction=U)


def read_effective_gf_iw(dir_='.', expand=False) -> SpinResolvedArray:
    """Return the effective atomic Green's function from file in `dir_`.

    The effective atomic Green's function is defined

    .. math:: F(z) = (z + μ - ϵ - Σ(z))^{-1}= [G_{imp}^{-1}(z) + Δ(z)]^{-1},

    with the hybridization function :math:`Δ(z)`.

    Parameters
    ----------
    dir_ : str or Path
        The directory where the output of the **layer_hb** code is located.
    expand : bool, optional
        Performs the symmetry expansion to return data for all layers.
        See `expand_layers`.

    Returns
    -------
    effective_gf_iw : (2, N_l, N_iw) util.SpinResolvedArray
        The effective atomic Green's function. The shape of the array is
        (# spins, # layers, # Matsubara frequencies).

    """
    out_dir = output_dir(dir_)
    dmft_output = np.loadtxt(out_dir / DMFT_FILE, unpack=True)
    effective_gf_iw_real = dmft_output[5::STEP]
    effective_gf_iw_imag = dmft_output[6::STEP]
    effective_gf_iw = effective_gf_iw_real + 1j*effective_gf_iw_imag
    effective_gf_iw = SpinResolvedArray(
        up=effective_gf_iw[0::2],
        dn=effective_gf_iw[1::2]
    )
    if expand:
        return effective_gf_iw[expand_layers(axis=1, dir_=dir_)]
    return effective_gf_iw


def read_occ(dir_='.') -> SpinResolvedArray:
    """Return the layer resolved occupation from file in `dir_`.

    Parameters
    ----------
    dir_ : str or Path
        The directory where the output of the **layer_hb** code is located.

    Returns
    -------
    occ.x : (2, N_l) util.SpinResolvedArray
        The layer resolved occupation. The shape of the array is
        (# spins, # layers).
    occ.err : (2, N_l) util.SpinResolvedArray
        The statistical error corresponding to `occ.x` from Monte Carlo.

    """
    out_dir = output_dir(dir_)
    layer_output = np.loadtxt(out_dir / OCC_FILE, unpack=True)
    # 0: layer 1: up 2: up_err 3: dn 4: dn_err
    occ = SpinResolvedArray(
        up=layer_output[1],
        dn=layer_output[3]
    )
    occ_err = SpinResolvedArray(
        up=layer_output[2],
        dn=layer_output[4]
    )
    return gt.Result(x=occ, err=occ_err)
