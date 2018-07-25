"""Collection of standard plotting functions for this module."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from builtins import (ascii, bytes, chr, dict, filter, hex, input, int, map,
                      next, oct, open, pow, range, round, str, super, zip)
from itertools import cycle

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from wrapt import decorator

FILLED_MARKERS = cycle(('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'))
DEFAULT_MARKER = 'x'


@decorator
def default_axis(wrapped, instance, args, kwargs):
    """Use current axis if axis is not explicitly given."""
    def _wrapper(*args, axis=None, **kwargs):
        if axis is None:
            axis = plt.gca()
        return wrapped(*args, axis=axis, **kwargs)
    return _wrapper(*args, **kwargs)


@default_axis
def V_data(V_l, axis=None, **mpl_args):
    """Plot default graph for potential `V_l`.

    Parameters
    ----------
    V_l : ndarray(float)
        The data of the Coulomb potential.
    axis : matplotlib axis, optional
        Axis on which the plot will be drawn. If `None` current one is used.
    **mpl_args :
        Arguments passed to `mpl.plot`

    """
    default_style = {
        'marker': DEFAULT_MARKER,
        'color': 'black',
        'linestyle': '--',
    }
    # default_style.update(mpl_args)
    axis.plot(V_l, **default_style)
    axis.set_ylabel(r'$V_l$')
    axis.set_xlabel('layer')


@default_axis
def V(param, layer_max=None, axis=None,
           label_str='{i}\n'
                     '$h={{param.h[{i}]:+.2f}}$\n'
                     '$\\mu={{param.mu[{i}]:+.2f}}$\n'
                     # '$V={{param.V[{i}]:+.2f}}$\n'
                     '$U={{param.U[{i}]:+.2f}}$',
           label_short='{i}\n'
                       '{{param.h[{i}]:+.2f}}\n'
                       '{{param.mu[{i}]:+.2f}}\n'
                       # '{{param.V[{i}]:+.2f}}\n'
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
    label_str : str
        The template string for the y-labels. **i** is replaced with the layer
        number. **param** can be used to print parameters of the calculation.
        If **label_str** is not `None`, this will be just printed for the first
        layer and consecutive layers will use **label_str** instead.
    label_short : str, None
        If `label_short` is not none, it will be used for all labels from the
        second layer on. See **label_str**.

    """
    V_l = param.V
    layers = np.arange(V_l.size)  # FIXME: not used
    if isinstance(layer_max, int):
        layers = layers[:layer_max]
    elif isinstance(layer_max, slice):
        layers = layers[layer_max]
    elif layer_max is not None:
        raise TypeError("unsupported type for `layer_max`: {}".format(type(layer_max)))

    axis.axhline(y=0, color='darkgray')
    V_data(V_l, axis=axis, **mpl_args)

    # TODO: do more sophisticated algorithm, checking for homogeneous regions
    if layers.size > 10:
        layers = layers[::layers.size//10]
    if label_short is None:
        labels = [label_str.format(i=i).format(param=param) for i in layers]
    else:
        labels = [label_short.format(i=i).format(param=param) for i in layers[1:]]
        labels.insert(0, label_str.format(i=layers[0]).format(param=param))
    axis.set_xticks(layers)
    axis.set_xticklabels(labels)
    axis.set_xlim(left=layers[0]-.5, right=layers[-1]+.5)
    axis.yaxis.set_minor_locator(AutoMinorLocator())
    # axis.minorticks_on()
    axis.grid(b=True)

    if np.all(V_l >= 0.):
        axis.set_ylim(bottom=0.)
    elif np.all(V_l <= 0.):
        axis.set_ylim(top=0.)


@default_axis
def n_data(n_l, spin='both', axis=None, **mpl_args):
    """Plot default graph for occupation `n_l`.

    Parameters
    ----------
    n_l : ndarray(float)
        The data of the occupation. The expected shape is (2, layers).
    spin : {'up', 'dn', 'both', 'sum'}
        Which spin channel to plot. `n_l[0]` corresponds to up and `n_l[1]` to
        down.
    axis : matplotlib axis, optional
        Axis on which the plot will be drawn. If `None` current one is used.
    **mpl_args :
        Arguments passed to `mpl.plot`

    """
    assert spin in set(('up', 'dn', 'both', 'sum'))
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
        'up': n_l[0],
        'dn': n_l[1],
        'sum': n_l.sum(axis=0),
    }

    def _plot_spin(spin):
        default_style['marker'] = marker[spin]
        default_style.update(mpl_args)
        axis.plot(data[spin], **default_style)

    if spin == 'both':
        for sp in ('up', 'dn'):
            _plot_spin(sp)
    else:
        _plot_spin(spin)

    axis.set_ylabel(r'$n_l$')
    axis.set_xlabel('layer')


@default_axis
def magnetisation_data(n_l, axis=None, **mpl_args):
    """Plot default graph for the magnetization :math:`n_↑ - n_↓`.

    Parameters
    ----------
    n_l : ndarray(float)
        The data of the occupation. The expected shape is (2, layers).
    axis : matplotlib axis, optional
        Axis on which the plot will be drawn. If `None` current one is used.
    **mpl_args :
        Arguments passed to `mpl.plot`

    """
    default_style = {
        'color': 'black',
        'linestyle': '--',
        'marker': DEFAULT_MARKER,
    }
    default_style.update(mpl_args)
    axis.plot(n_l[0] - n_l[1], **default_style)

    axis.set_ylabel(r'$n_{l\uparrow} - n_{l\downarrow}$')
    axis.set_xlabel('layer')
