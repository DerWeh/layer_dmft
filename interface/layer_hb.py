#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : layer_hb.py
# Author            : Weh Andreas <andreas.weh@physik.uni-augsburg.de>
# Date              : 31.08.2018
# Last Modified Date: 23.10.2018
# Last Modified By  : Weh Andreas <andreas.weh@physik.uni-augsburg.de>
"""Utilities to interact with Junya's **layer_hb** code for R-DMFT.

This module contains function to read the output, and run the code.
"""
from __future__ import division, print_function, unicode_literals

from pathlib import Path

import numpy as np

# FIXME
from sys import path as syspath

_PATH = Path(__file__).absolute()
syspath.insert(1, str(_PATH.parents[1]))

import model
import gftools as gt
from gftools import pade as gtpade

# from .. import model


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


class SelfEnergy(model.SpinResolvedArray):
    """`ndarray` wrapper for self-energies for the Hubbard model within DMFT."""
    def __new__(cls, input_array, occupation, interaction):
        """Create `SelfEnergy` from existing array_like input.

        Adds capabilities to separate the static Hartree part from the self-
        energy.

        Parameters
        ----------
        input_array : (N_s, N_l, [N_w]) array_like
            Date points for the self-energy, N_s is the number of spins,
            N_l the number of layers and N_w the number of frequencies.
        occupation : (N_s, N_l) array_like
            The corresponding occupation numbers, to calculate the moments.
        interaction : (N_l, ) array_like
            The interaction strength Hubbard :math:`U`.

        """
        obj = np.asarray(input_array).view(cls)
        obj._N_s, obj._N_l = obj.shape[:2]  # #Spins, #Layers
        obj.occupation = np.asanyarray(occupation)
        assert obj.occupation.shape == (obj._N_s, obj._N_l)
        obj.interaction = np.asarray(interaction)
        assert obj.interaction.shape == (obj._N_l, )
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.occupation = getattr(obj, 'occupation', None)
        self.interaction = getattr(obj, 'interaction', None)
        self._N_s = getattr(obj, '_N_s', None)
        self._N_l = getattr(obj, '_N_l', None)

    def dynamic(self):
        """Return the dynamic part of the self-energy.

        The static mean-field part is stripped.

        Returns
        -------
        dynamic : (N_s, N_l, [N_w]) ndarray
            The dynamic part of the self-energy

        Raises
        ------
        IndexError
            If the shape of the data doesn't match `occupation` and `interaction`
            shape.

        """
        self: np.ndarray
        if self.shape[:2] != (self._N_s, self._N_l):
            raise IndexError(f"Mismatch of data shape {self.shape} and "
                             f"additional information ({self._N_s}, {self._N_l})"
                             "\n Slicing is not implemented to work with the methods")
        static = self.static(expand=True)
        dynamic = self.view(type=np.ndarray) - static
        # try:
        #     return self - static[..., np.newaxis]
        # except ValueError as val_err:  # if N_w axis doesn't exist
        #     if len(self.shape) != 2:
        #         raise val_err
        #     return self - static
        return dynamic

    def static(self, expand=False):
        """Returns the static (Hartree mean-field) part of the self-energy.

        If `expand`, the dimension for `N_w` is added"""
        static = self.occupation[::-1] * self.interaction
        if expand and len(self.shape) == 3:
            static = static[..., np.newaxis]
        return static

    def pade(self, z_out, z_in, n_min: int, n_max: int, valid_z, threshold=1e-8):
        """Perform Pade analytic continuation on the self-energy.

        Parameters
        ----------
        z_out : complex or array_like
            Frequencies at which the continuation will be calculated.
        z_in : complex ndarray
            Frequencies corresponding to the input self-energy `self`.
        n_min, n_max : int
            Minimum/Maximum number of frequencies considered for the averaging.

        Returns
        -------
        pade.x : (N_s, N_l, N_z_out) ndarray
            Analytic continuation. N_s is the number of spins, N_l the number
            of layer, and N_z_out correspond to `z_in.size`.
        pade.err : (N_s, N_l, N_z_out) ndarray
            The variance corresponding to `pade.x`.

        """
        z_out = np.asarray(z_out)
        pade_fct = np.vectorize(
            lambda self_sl:
            gtpade.averaged(z_out, z_in, gf_iw=self_sl, n_min=n_min, n_max=n_max,
                            valid_z=valid_z, threshold=threshold, kind='self'),
            otypes=(np.complex, np.complex),
            doc=gtpade.averaged.__doc__,
            signature='(n)->(m),(m)'
        )
        # Pade performs better if static part is not stripped from self-energy
        # # static part needs to be stripped as function is for Gf not self-energy
        # self_pade, self_pade_err = pade_fct(self.dynamic())
        self_pade, self_pade_err = pade_fct(self)
        self_pade = self_pade.squeeze()
        self_pade_err = self_pade_err.squeeze()
        # return gt.Result(x=self_pade+self.static(expand=True), err=self_pade_err)
        return gt.Result(x=self_pade, err=self_pade_err)


def output_dir(dir_):
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
        raise ValueError('{} not in stream {}'.format(word, fb))
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


def load_param(dir_='.'):
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


def read_gf_iw(dir_='.', expand=False):
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
    gf_iw : (2, N_l, N_iw) model.SpinResolvedArray
        The local Green's function. The shape of the array is
        (#spins, #layers, #Matsubara frequencies).

    """
    out_dir = output_dir(dir_)
    gf_output = np.loadtxt(out_dir / GF_FILE, unpack=True)
    gf_iw_real = gf_output[1::IM_STEP]
    gf_iw_imag = gf_output[2::IM_STEP]
    gf_iw = gf_iw_real + 1j*gf_iw_imag
    gf_iw = model.SpinResolvedArray(
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
    self_iw : (2, N_l, N_iw) model.SpinResolvedArray
        The self-energy. The shape of the array is
        (#spins, #layers, #Matsubara frequencies).

    """
    out_dir = output_dir(dir_)
    gf_output = np.loadtxt(out_dir / SELF_FILE, unpack=True)
    self_real = gf_output[1::IM_STEP]
    self_imag = gf_output[2::IM_STEP]
    self = self_real + 1j*self_imag
    self = model.SpinResolvedArray(
        up=self[0::2],
        dn=self[1::2]
    )
    prm = load_param(dir_)
    occ, __ = read_occ(dir_)
    if expand:
        self = self[expand_layers(axis=1, dir_=dir_)]
    #     self_static = occ[::-1] * prm.U
    # else:
    #     self_static = (occ[::-1] * prm.U)[reduce_layers(axis=1, dir_=dir_)]
    # self += self_static[..., np.newaxis]
    self += .5 * prm.U[:, np.newaxis]
    return SelfEnergy(self, occupation=occ, interaction=prm.U)


def read_effective_gf_iw(dir_='.', expand=False):
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
    effective_gf_iw : (2, N_l, N_iw) model.SpinResolvedArray
        The effective atomic Green's function. The shape of the array is
        (# spins, # layers, # Matsubara frequencies).

    """
    out_dir = output_dir(dir_)
    dmft_output = np.loadtxt(out_dir / DMFT_FILE, unpack=True)
    effective_gf_iw_real = dmft_output[5::STEP]
    effective_gf_iw_imag = dmft_output[6::STEP]
    effective_gf_iw = effective_gf_iw_real + 1j*effective_gf_iw_imag
    effective_gf_iw = model.SpinResolvedArray(
        up=effective_gf_iw[0::2],
        dn=effective_gf_iw[1::2]
    )
    if expand:
        return effective_gf_iw[expand_layers(axis=1, dir_=dir_)]
    return effective_gf_iw


def read_occ(dir_='.'):
    """Return the layer resolved occupation from file in `dir_`.

    Parameters
    ----------
    dir_ : str or Path
        The directory where the output of the **layer_hb** code is located.

    Returns
    -------
    occ.x : (2, N_l) model.SpinResolvedArray
        The layer resolved occupation. The shape of the array is
        (# spins, # layers).
    occ.err : (2, N_l) model.SpinResolvedArray
        The statistical error corresponding to `occ.x` from Monte Carlo.

    """
    out_dir = output_dir(dir_)
    layer_output = np.loadtxt(out_dir / OCC_FILE, unpack=True)
    # 0: layer 1: up 2: up_err 3: dn 4: dn_err
    occ = model.SpinResolvedArray(
        up=layer_output[1],
        dn=layer_output[3]
    )
    occ_err = model.SpinResolvedArray(
        up=layer_output[2],
        dn=layer_output[4]
    )
    return gt.Result(x=occ, err=occ_err)
