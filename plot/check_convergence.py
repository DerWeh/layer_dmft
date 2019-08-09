"""Visualize convergence."""
from functools import reduce
from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm

from layer_dmft import layer_dmft, util, dataio, plot

with util.local_import():
    from init import prm

LAY_OUT = 'layer_output'
IMP_OUT = 'imp_output'

# TODO: what if init overwrites `FORCE_PARAMAGNET`?
PARAMAGNETIC = False if np.count_nonzero(prm.h) or not layer_dmft.FORCE_PARAMAGNET else True
BARE_OCCUPATIONS = True

# import matplotlib as mpl
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

MARKERS = {'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'}


def get_difference_figure(fig_name):
    if PARAMAGNETIC:
        plt.figure(num=fig_name)
        axes = (plt.gca(),)
    else:
        __, axes = plt.subplots(2, sharex=True, sharey=True, num=fig_name)

    for axis in axes:
        axis.axhline(y=0., color='black', linewidth=.3)
    return axes


lay_data = dataio.LayerData(dir_=LAY_OUT)
imp_data = dataio.ImpurityData(dir_=IMP_OUT)


different_iterations = lay_data.iterations ^ imp_data.iterations
if different_iterations and different_iterations != {-1}:
    print("Missmachting iteration data XXX")
    print(different_iterations)

iterations = sorted(lay_data.iterations & imp_data.iterations)
layer_per_iter = [list(imp_data.iter(it).keys()) for it in iterations]
layers = reduce(set.union, layer_per_iter[1:], set(layer_per_iter[0]))

# filter out 'not calculate' layers
gf_lay_iw_iter = [gf_iw_iter[:, lays] for lays, gf_iw_iter in zip(layer_per_iter, lay_data.gf_iw)]
gf_imp_iw_iter = [np.array([imp_data.iter(it)[lay]['gf_iw'] for lay in layers]).transpose(1, 0, 2)
                  # layers are second index
                  for layers, it in zip(layer_per_iter, iterations)]
total_distance = [norm(gf_lay_iw - gf_imp_iw)/norm(gf_lay_iw)
                  for gf_lay_iw, gf_imp_iw in zip(gf_lay_iw_iter, gf_imp_iw_iter)]

#
# Absolute distance between layer and impurity Green's function
#
plt.figure('Distance')
plt.plot(iterations, total_distance, 'D--', label='all layers')

marker_cycle = cycle(MARKERS - {'D'})
for lay in layers:
    lay_iterations = [it for it, layers in zip(iterations, layer_per_iter) if lay in layers]
    lay_indices = [iterations.index(it) for it in lay_iterations]
    gf_lay_iw_lay = lay_data.gf_iw[lay_indices, :, lay]
    gf_imp_iw_lay = np.array([imp_data.iter(it)[lay]['gf_iw'] for it in lay_iterations])
    lay_distance = [norm(gf_lay_iw_lay_it - gf_imp_iw_lay_it)/norm(gf_lay_iw_lay_it) for
                    gf_lay_iw_lay_it, gf_imp_iw_lay_it in zip(gf_lay_iw_lay, gf_imp_iw_lay)]
    plt.plot(lay_iterations, lay_distance, ':', label=str(lay), marker=next(marker_cycle))

plt.ylabel(r'$||G_{\mathrm{lay}}(iω_n) - G_{\mathrm{imp}}(iω_n)||/||G_{\mathrm{lay}}(iω_n)||$')
plt.yscale('log')
plt.xlabel('iteration')
plt.legend()
plt.tight_layout()


#
# Distance between imaginary part of iω_0 of layer and impurity Green's function
#
axes = get_difference_figure('Difference iω_0')
sp_text = ('↑', '↓')


marker_cycle = cycle(MARKERS - {'D'})
for lay in layers:
    lay_iterations = [it for it, layers in zip(iterations, layer_per_iter) if lay in layers]
    lay_indices = [iterations.index(it) for it in lay_iterations]
    gf_lay_iw0_lay = lay_data.gf_iw[lay_indices, :, lay, 0].imag
    gf_imp_iw0_lay = np.array([imp_data.iter(it)[lay]['gf_iw'][:, 0].imag for it in lay_iterations])
    if PARAMAGNETIC:
        gf_lay_iw0_lay = gf_lay_iw0_lay.mean(axis=-1)
        gf_imp_iw0_lay = gf_imp_iw0_lay.mean(axis=-1)
        plt.plot(lay_iterations, (gf_lay_iw0_lay-gf_imp_iw0_lay)/gf_lay_iw0_lay,
                 ':', label=str(lay), marker=next(marker_cycle))
    else:
        diff = (gf_lay_iw0_lay-gf_imp_iw0_lay)/gf_lay_iw0_lay
        for sp in (0, 1):
            axes[sp].plot(lay_iterations, diff[:, sp], ':', label=str(lay),
                          marker=next(marker_cycle))

for sp, axis in enumerate(axes):
    axis.set_ylabel(r'$\Im(G_{\mathrm{lay}}(iω_0) - G_{\mathrm{imp}}(iω_0))'
                    r'/\Im G_{\mathrm{lay}}(iω_0)$'
                    + (f' $(σ={sp_text[sp]})$' if not PARAMAGNETIC else ''))
    # axis.set_yscale('symlog')
    axis.set_xlabel('iteration')
    axis.legend()
plt.tight_layout()


#
# Distance of occupation
#
axes = get_difference_figure('Difference n_l')
if BARE_OCCUPATIONS:
    axes_ = get_difference_figure('n_l')
sp_text = ('↑', '↓')

marker_cycle = cycle(MARKERS - {'D'})
for lay in layers:
    lay_iterations = [it for it, layers in zip(iterations, layer_per_iter) if lay in layers]
    lay_indices = [iterations.index(it) for it in lay_iterations]
    occ_imp = np.array([-imp_data.iter(it)[lay]['gf_tau'][:, -1] for it in lay_iterations])
    occ_imp_err = np.array([imp_data.iter(it)[lay]['gf_tau_err'][:, -1] for it in lay_iterations])
    gf_lay = lay_data.gf_iw
    occ_lay = prm.occ0(gf_lay, hartree=lay_data.occ[:, ::-1], return_err=True)
    occ_lay, occ_lay_err = occ_lay.x[lay_indices, ..., lay], occ_lay.err[lay_indices, ..., lay]
    if PARAMAGNETIC:
        occ_lay = occ_lay.mean(axis=-1)
        occ_imp = occ_imp.mean(axis=-1)
        # might be inaccurate but doesn't really matter
        occ_imp_err = occ_imp_err.mean(axis=-1)
        occ_lay_err = occ_lay_err.mean(axis=-1)
        err = abs(occ_imp_err/occ_lay) + abs(occ_lay_err*occ_imp/occ_lay**2)
        plot.err_plot(lay_iterations, (occ_lay-occ_imp)/occ_lay, yerr=err, axis=axes[0],
                      linestyle=':', label=str(lay), marker=next(marker_cycle))
    else:
        diff = (occ_lay-occ_imp)/occ_lay
        err = abs(occ_imp_err/occ_lay) + abs(occ_lay_err*occ_imp/occ_lay**2)
        for sp in (0, -1):
            plot.err_plot(lay_iterations, diff[:, sp], yerr=err[:, sp], axis=axes[sp],
                          linestyle=':', label=str(lay), marker=next(marker_cycle))
    if BARE_OCCUPATIONS:
        if PARAMAGNETIC:
            # might be inaccurate but doesn't really matter
            plot.err_plot(lay_iterations, occ_lay, yerr=occ_lay_err, axis=axes_[0],
                          linestyle=':', label='lay'+str(lay), marker=next(marker_cycle))
            plot.err_plot(lay_iterations, occ_imp, yerr=occ_imp_err, axis=axes_[0],
                          linestyle=':', label='imp'+str(lay), marker=next(marker_cycle))
        else:
            for sp in (0, -1):
                plot.err_plot(lay_iterations, occ_lay[:, sp], yerr=occ_lay_err[:, sp], axis=axes_[sp],
                              linestyle=':', label='lay'+str(lay), marker=next(marker_cycle))
                plot.err_plot(lay_iterations, occ_imp[:, sp], yerr=occ_imp_err[:, sp], axis=axes_[sp],
                              linestyle=':', label='imp'+str(lay), marker=next(marker_cycle))



for sp, axis in enumerate(axes):
    axis.set_ylabel(r'$(n_{\mathrm{lay}} - n_{\mathrm{imp}})'
                    r'/ n_{\mathrm{lay}}$'
                    + (f' $(σ={sp_text[sp]})$' if not PARAMAGNETIC else ''))
    # axis.set_yscale('symlog')
    axis.set_xlabel('iteration')
    axis.legend()
if BARE_OCCUPATIONS:
    for sp, axis in enumerate(axes_):
        axis.set_ylabel(r'$n_l$' + (f' $(σ={sp_text[sp]})$' if not PARAMAGNETIC else ''))
        # axis.set_yscale('symlog')
        axis.set_xlabel('iteration')
        axis.legend()
plt.tight_layout()

plt.show()
