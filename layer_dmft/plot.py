#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : plot.py
# Author            : Weh Andreas <andreas.weh@physik.uni-augsburg.de>
# Date              : 02.08.2018
# Last Modified Date: 23.10.2018
# Last Modified By  : Weh Andreas <andreas.weh@physik.uni-augsburg.de>
"""Collection of standard plotting functions for this module."""
from itertools import cycle
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_pdf import PdfPages

FILLED_MARKERS = cycle(('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'))
DEFAULT_MARKER = 'x'
ERR_CAPSIZE = 2


def err_plot(x, y, yerr=None, xerr=None, axis: mpl.axes.Axes = None, **mpl_args):
    """Plot graph with error bars.

    Decided weather to plot points with error bars or lines with shaded areas
    depending on the number of plotted points.

    Parameters
    ----------
    x, y : array_like
        x and y coordinates of the data to plot.
    yerr, xerr : array_like, optional
        The corresponding error of the data. Has same shape `x` and `y`.
    axis : mpl.axes.Axes, optional
        `mpl.axes.Axes` object used for plotting.
    mpl_args :
        Arguments for plotting passed to `axis.errorbar` or `axis.plot`.

    """
    axis = plt.gca() if axis is None else axis
    x = np.asarray(x)
    if x.size > 50:  # continuous plot
        try:
            ecolor = mpl_args.pop('ecolor')
        except KeyError:  # no color defined -> try color else default
            ecolor = mpl_args.get('color', None)
        try:
            fmt = mpl_args.pop('fmt')
        except KeyError:
            baseline, = axis.plot(x, y, **mpl_args)
        else:
            baseline, = axis.plot(x, y, fmt, **mpl_args)
        if ecolor is None:
            ecolor = baseline.get_color()
        if yerr is not None:
            axis.fill_between(x, y-yerr, y+yerr, color=ecolor, alpha=.3, zorder=1)
        if xerr is not None:
            axis.fill_betweenx(y, x-xerr, x+xerr, color=ecolor, alpha=.3, zorder=1)
    else:
        default_args = {'capsize': 2.5, 'elinewidth': .3}
        default_args.update(mpl_args)
        axis.errorbar(x, y, yerr=yerr, **default_args)


@contextmanager
def print_param(filename, param, **kwds):
    """Context manager to save plots with the used parameters on a separate page.

    `PdfPages.savefig` is used to write a multi page PDF, on the last page
    the parameters `param` are printed.
    """
    # add git SHA <- version
    print_file = Path(filename)
    if print_file.suffix.lower() != '.pdf':
        print_file = print_file.parent / (print_file.name + '.pdf')
    filename = str(print_file)

    with PdfPages(filename, **kwds) as pdf:
        yield pdf
        pdf.savefig()
        plt.close(plt.gcf())

        fig, ax = plt.subplots()
        ax.set_title('Hubbard parameters')
        ax.axison = False
        ax.text(0.05, 0.95, str(param),
                verticalalignment='top', horizontalalignment='left',
                transform=ax.transAxes)
        fig.tight_layout()
        pdf.attach_note("Hubbard model parameters")
        pdf.savefig(fig)
        plt.close(fig)


def V_data(V_l, axis=None, **mpl_args):
    """Plot default graph for potential `V_l`.

    Parameters
    ----------
    V_l : ndarray(float)
        The data of the Coulomb potential.
    axis : mpl.axes.Axes, optional
        `mpl.axes.Axes` object used for plotting.
    mpl_args :
        Arguments passed to `mpl.plot`

    """
    axis = plt.gca() if axis is None else axis
    default_style = {
        'marker': DEFAULT_MARKER,
        'color': 'black',
        'linestyle': '--',
    }
    default_style.update(mpl_args)
    axis.plot(V_l, **default_style)
    axis.set_ylabel(r'$V_l$')
    axis.set_xlabel('layer')


def V(param, layer_max=None, axis=None,
      label_str='{i}\n'
                '$h={{param.h[{i}]:+.2f}}$\n'
                '$\\mu={{param.mu[{i}]:+.2f}}$\n'
                '$U={{param.U[{i}]:+.2f}}$',
      label_short='{i}\n'
                  '{{param.h[{i}]:+.2f}}\n'
                  '{{param.mu[{i}]:+.2f}}\n'
                  '{{param.U[{i}]:+.2f}}',
      **mpl_args):
    """Plot the Coulomb potential of the `param`.

    Parameters
    ----------
    param : `model.prm`
        `prm` object with the parameters set.
    layer_max : int or slice, optional
        The maximum layer number, up to which the occupation is plotted.
        Or slice specifying the range of plotted layers.
    axis : mpl.axes.Axes, optional
        `mpl.axes.Axes` object used for plotting.
    label_str : str
        The template string for the y-labels. **i** is replaced with the layer
        number. **param** can be used to print parameters of the calculation.
        If `label_str` is not `None`, this will be just printed for the first
        layer and consecutive layers will use `label_str` instead.
    label_short : str, None
        If `label_short` is not none, it will be used for all labels from the
        second layer on. See `label_str`.

    Raises
    ------
    TypeError
        If `layer_max` is **not** `int` or `slice`.

    """
    axis = plt.gca() if axis is None else axis
    V_l = param.V
    layers = np.arange(V_l.size)
    if isinstance(layer_max, int):
        layers = layers[:layer_max]
    elif isinstance(layer_max, slice):
        layers = layers[layer_max]
    elif layer_max is not None:
        raise TypeError(f"unsupported type for `layer_max`: {type(layer_max)}")

    axis.axhline(y=0, color='darkgray')
    V_data(V_l, axis=axis, **mpl_args)

    # TODO: do more sophisticated algorithm, checking for homogeneous regions
    if layers.size > 10:
        labeled_layers = layers[::layers.size//10]
    if label_short is None:
        labels = [label_str.format(i=i).format(param=param) for i in labeled_layers]
    else:
        labels = [label_short.format(i=i).format(param=param) for i in labeled_layers[1:]]
        labels.insert(0, label_str.format(i=layers[0]).format(param=param))
    axis.set_xticks(labeled_layers)
    axis.set_xticklabels(labels)
    axis.set_xlim(left=layers[0]-.5, right=layers[-1]+.5)
    axis.yaxis.set_minor_locator(AutoMinorLocator())
    # axis.minorticks_on()
    axis.grid(b=True)

    if np.all(V_l >= 0.):
        axis.set_ylim(bottom=0.)
    elif np.all(V_l <= 0.):
        axis.set_ylim(top=0.)


def _contains_error(occ):
    """Check if `occ` is only the occupation or contains the corresponding error."""
    if isinstance(occ, tuple):
        assert len(occ) == 2, "Must be tuple (values, errors)"
        assert occ[0].shape[0] == 2, "Values must be ndarray(up, dn)"
        error = True
    else:
        assert occ.shape[0] == 2, "Values must be ndarray(up, dn)"
        error = False
    return error


def occ(occ, spin='both', axis=None, **mpl_args):
    """Plot default graph for occupation `occ`, possibly with errorbars.

    This graph is designed to work with the output of `gftools.density`.
    If `occ` is a tuple, it is assumed that `occ[1]` contains the error.

    Parameters
    ----------
    occ : float ndarray or tuple of ndarray
        The data of the occupation. The expected shape is (2, layers).
        Alternative a tuple of two corresponding arrays can be given, were the
        second element is the error estimate.
    spin : {'up', 'dn', 'both', 'sum'}
        Which spin channel to plot. `occ[0]` corresponds to up and `occ[1]` to
        down.
    axis : mpl.axes.Axes, optional
        `mpl.axes.Axes` object used for plotting.
    mpl_args :
        Arguments passed to `mpl.plot`

    """
    axis = plt.gca() if axis is None else axis
    assert spin in set(('up', 'dn', 'both', 'sum'))
    error = _contains_error(occ)
    if error:
        occ_data_err(occ[0], occ_err=occ[1], spin=spin, axis=axis, **mpl_args)
    else:
        occ_data(occ, spin=spin, axis=axis, **mpl_args)


def occ_data(occ, spin='both', axis=None, **mpl_args):
    """Plot default graph for occupation `occ` data.

    This graph is designed to work with the output of `gftools.density` without
    errors.

    Parameters
    ----------
    occ : ndarray(float)
        The data of the occupation. The expected shape is (2, layers).
    spin : {'up', 'dn', 'both', 'sum'}
        Which spin channel to plot. `occ[0]` corresponds to up and `occ[1]` to
        down.
    axis : mpl.axes.Axes, optional
        `mpl.axes.Axes` object used for plotting.
    mpl_args :
        Arguments passed to `mpl.plot`

    """
    assert spin in set(('up', 'dn', 'both', 'sum'))
    axis = plt.gca() if axis is None else axis
    marker = {
        'up': '^',
        'dn': 'v',
        'sum': DEFAULT_MARKER,
    }
    default_style = {
        # 'color': 'black',
        'linestyle': '--',
    }

    data = {
        'up': occ[0],
        'dn': occ[1],
        'sum': occ.sum(axis=0),
    }

    def _plot_spin(spin):
        default_style['marker'] = marker[spin]
        default_style['label'] = 'n_' + spin
        default_style.update(mpl_args)
        axis.plot(data[spin], **default_style)

    if spin == 'both':
        for sp in ('up', 'dn'):
            _plot_spin(sp)
        axis.legend()
    else:
        _plot_spin(spin)

    axis.set_ylabel(r'$n_l$')
    axis.set_xlabel('layer')


def occ_data_err(occ, occ_err, spin='both', axis=None, **mpl_args):
    """Plot default graph for occupation `occ` data with error.

    This graph is designed to work with the output of `gftools.density` with
    errors.

    Parameters
    ----------
    occ, occ_err : (2, #layers) float np.ndarray
        The data and error of the occupation.
    spin : {'up', 'dn', 'both', 'sum'}
        Which spin channel to plot. `occ[0]` corresponds to up and `occ[1]` to
        down.
    axis : mpl.axes.Axes, optional
        `mpl.axes.Axes` object used for plotting.
    mpl_args :
        Arguments passed to `mpl.plot`

    """
    assert spin in set(('up', 'dn', 'both', 'sum'))
    axis = plt.gca() if axis is None else axis
    layers = np.arange(occ[0].size)
    marker = {
        'up': '^',
        'dn': 'v',
        'sum': DEFAULT_MARKER,
    }
    default_style = {
        # 'color': 'black',
        'linestyle': '--',
        'capsize': ERR_CAPSIZE,
    }

    def _plot_dict(_occ, _occ_err):
        return {'y': _occ, 'yerr': _occ_err}

    data = {
        'up': _plot_dict(occ[0], occ_err[0]),
        'dn': _plot_dict(occ[1], occ_err[1]),
        'sum': _plot_dict(occ.sum(axis=0), occ_err.sum(axis=0)),
    }

    def _plot_spin(spin):
        default_style['marker'] = marker[spin]
        default_style['label'] = 'n_' + (spin)
        default_style.update(mpl_args)
        axis.errorbar(x=layers, **data[spin], **default_style)

    if spin == 'both':
        for sp in ('up', 'dn'):
            _plot_spin(sp)
        axis.legend()
    else:
        _plot_spin(spin)

    axis.set_ylabel(r'$n_l$')
    axis.set_xlabel('layer')


def magnetization(occ, axis=None, **mpl_args):
    """Plot the magnetization :math:`n_↑ - n_↓`, possibly with errorbars.

    Parameters
    ----------
    occ : float ndarray or tuple of ndarray
        The data of the occupation. The expected shape is (2, layers).
        Alternative a tuple of two corresponding arrays can be given, were the
        second element is the error estimate.
    axis : mpl.axes.Axes, optional
        `mpl.axes.Axes` object used for plotting.
    mpl_args :
        Arguments passed to `mpl.plot`

    """
    axis = plt.gca() if axis is None else axis
    error = _contains_error(occ)
    if error:
        magnetization_data_error(occ[0], occ_err=occ[1], axis=axis, **mpl_args)
    else:
        magnetization_data(occ, axis=axis, **mpl_args)


def magnetization_data(occ, axis=None, **mpl_args):
    """Plot default graph for the magnetization :math:`n_↑ - n_↓`.

    Parameters
    ----------
    occ : (2, layers) float ndarray
        The data of the occupation.
    axis : mpl.axes.Axes, optional
        `mpl.axes.Axes` object used for plotting.

    """
    axis = plt.gca() if axis is None else axis
    default_style = {
        'color': 'black',
        'linestyle': '--',
        'marker': DEFAULT_MARKER,
    }
    default_style.update(mpl_args)
    axis.plot(occ[0] - occ[1], **default_style)

    axis.set_ylabel(r'$n_{l\uparrow} - n_{l\downarrow}$')
    axis.set_xlabel('layer')


def magnetization_data_error(occ, occ_err, axis=None, **mpl_args):
    """Plot default graph for the magnetization :math:`n_↑ - n_↓`.

    Parameters
    ----------
    occ, occ_err : (2, layers) float ndarray
        The data of the occupation and its error.
    axis : mpl.axes.Axes, optional
        `mpl.axes.Axes` object used for plotting.
    mpl_args :
        Arguments passed to `mpl.plot`

    """
    axis = plt.gca() if axis is None else axis
    layers = np.arange(occ[0].size)
    default_style = {
        'color': 'black',
        'linestyle': '--',
        'marker': DEFAULT_MARKER,
        'capsize': ERR_CAPSIZE,
    }
    default_style.update(mpl_args)
    axis.errorbar(x=layers, y=occ[0] - occ[1], yerr=occ_err[0] + occ_err[1],
                  **default_style)

    axis.set_ylabel(r'$n_{l\uparrow} - n_{l\downarrow}$')
    axis.set_xlabel('layer')


def hopping_matrix(t_mat, axis=None, log=False, **mpl_args):
    """Plot color representation of the hopping_matrix `t_mat`.

    The values are shown as color and the values are indexed.

    Parameters
    ----------
    t_mat : float ndarray
        The matrix containing the hopping elements.
    axis : mpl.axes.Axes, optional
        `mpl.axes.Axes` object used for plotting.
    log : bool, optional
        Weather the values are represented using a logarithmic scaling.
        Default is `False`.
    mpl_args :
        Arguments passed to `mpl.plot`

    """
    axis = plt.gca() if axis is None else axis
    if log:  # logarithmic plot
        norm = LogNorm(vmin=t_mat[t_mat > 0].min(), vmax=t_mat.max())
        imag = axis.matshow(t_mat, norm=norm, **mpl_args)
    else:
        imag = axis.matshow(t_mat, **mpl_args)
    cbar = axis.figure.colorbar(imag, ax=axis)

    # make white grid between matrix elements
    axis.set_xticks(np.arange(t_mat.shape[1]+1)-.5, minor=True)
    axis.set_yticks(np.arange(t_mat.shape[0]+1)-.5, minor=True)
    axis.grid(which="minor", color="w", linestyle='-')
    axis.tick_params(which="minor", top=False, left=False, bottom=False, right=False)

    # mark unique elements of t_mat at colorbar
    values, mapping = np.unique(t_mat, return_inverse=True)
    mapping = mapping.reshape(t_mat.shape)
    cbar.ax.yaxis.set_ticks(imag.norm(values), minor=True)
    cbar.ax.yaxis.set_ticklabels(np.arange(values.size), minor=True)
    cbar.ax.tick_params(axis='y', which='minor', direction='inout',
                        left='on', labelleft='on', labelright='off',
                        labelsize='x-large')

    threshold = None
    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = imag.norm(threshold)
    else:  # default to the half range
        threshold = imag.norm(t_mat.max())/2.

    # Set alignment to center
    kw = dict(horizontalalignment="center",
              verticalalignment="center")

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    color_index = (imag.norm(values) > threshold).view(np.ndarray)
    textcolors = ["white", "black"]
    for i, row in enumerate(mapping):
        for j, t_ij in enumerate(row):
            kw.update(color=textcolors[color_index[t_ij]])
            axis.text(j, i, str(t_ij), **kw)


def matrix(mat, axis=None, log=False, **mpl_args):
    """Plot color representation of a matrix `mat`.

    Parameters
    ----------
    mat : float ndarray
        The matrix containing the hopping elements.
    axis : mpl.axes.Axes, optional
        `mpl.axes.Axes` object used for plotting.
    log : bool, optional
        Weather the values are represented using a logarithmic scaling.
        Default is `False`.
    mpl_args :
        Arguments passed to `mpl.plot`

    """
    axis = plt.gca() if axis is None else axis
    if log:  # logarithmic plot
        norm = LogNorm(vmin=mat[mat > 0].min(), vmax=mat.max())
        imag = axis.matshow(mat, norm=norm, **mpl_args)
    else:
        imag = axis.matshow(mat, **mpl_args)
    axis.figure.colorbar(imag, ax=axis)

    # make white grid between matrix elements
    axis.set_xticks(np.arange(mat.shape[1]+1)-.5, minor=True)
    axis.set_yticks(np.arange(mat.shape[0]+1)-.5, minor=True)
    axis.grid(which="minor", color="w", linestyle='-')
    axis.tick_params(which="minor", top=False, left=False, bottom=False, right=False)
