"""Plot spectral function for real frequencies.

Options
-------
SPLIT_PLOTS
    Make a separate figure for every layer.
PARAMAGNETIC
    Plot only one spin channel (the average) as both are identical.
SHIFT
    The imaginary shift into the upper complex half plane for retarded
    Green's functions. This is a imaginary (complex variable) number,
    the imaginary part should be smaller than the first Matsubara frequency.
THRESHOLD
    Threshold, how positive the imaginary part of retarded quantities is
    allowed to get. This number should be smaller than the shift:
    `THRESHOLD < SHIFT.imag`.

"""
import logging

from functools import partial

import numpy as np
import matplotlib.pyplot as plt

import gftools as gt
from gftools import pade

from layer_dmft import util, layer_dmft, plot, dataio, model

#
# configure logging
#
prm: model.Hubbard_Parameters

with util.local_import():
    from init import prm

logging.basicConfig(level=logging.INFO)


#
# load data
#
lay_obj = dataio.LayerData()
iteration = tuple(lay_obj.iterations)[-1]
logging.debug("Load iteration %s of layer data", iteration)
lay_data = lay_obj.iter(iteration)

# FIXME: imp_data gets collected upon calling `iter` and discarding the object!
imp_obj = dataio.ImpurityData()
imp_data = imp_obj.iter(iteration)

assert lay_data["temperature"] == prm.T

layers = list(imp_data.keys())

gf_lay_iw: np.ndarray = lay_data['gf_iw']
gf_lay_iw = gf_lay_iw[:, layers]
gf_imp_iw = np.array([imp_lay['gf_iw'] for imp_lay in imp_data.values()]).transpose(1, 0, 2)
# gf_imp_iw = np.array([imp_data[lay]['gf_iw'] for lay in sorted(imp_data.keys())]).transpose(1, 0, 2)
# print(*sorted(imp_data.keys()))

N_iw = gf_lay_iw.shape[-1]
iws = gt.matsubara_frequencies(np.arange(N_iw), beta=prm.beta)

#
# options
#
SPLIT_PLOTS = False
PARAMAGNETIC = False if np.count_nonzero(prm.h) or not layer_dmft.FORCE_PARAMAGNET else True
SHIFT: complex = 0.2*iws[0]
# SHIFT *= 4  # FIXME
THRESHOLD: float = SHIFT.imag/2
THRESHOLD = np.infty  # FIXME


if PARAMAGNETIC:  # average over spins
    gf_lay_iw = gf_lay_iw.mean(axis=0, keepdims=True)  # should already be averaged
    gf_imp_iw = gf_imp_iw.mean(axis=0, keepdims=True)

omega = np.linspace(-4*prm.D, 4*prm.D, num=1000, dtype=complex)
omega += SHIFT  # shift into imaginary plane
N_w = omega.size

# prepare Pade
kind_gf = pade.KindGf(N_iw//10, 8*N_iw//10)
no_neg_imag = pade.FilterNegImag(THRESHOLD)
pade_gf = partial(pade.avg_no_neg_imag, z_in=iws, z_out=omega, kind=kind_gf, threshold=THRESHOLD)

# perform Pade
gf_lay_w = pade_gf(fct_z=gf_lay_iw)
gf_imp_w = pade_gf(fct_z=gf_imp_iw)

#
# create figure
#
rows, cols = gf_lay_iw.shape[:-1]
if SPLIT_PLOTS:
    axes = np.array([plt.subplots(nrows=rows, sharex=True, squeeze=False)[1][:, 0]
                     for __ in range(cols)]).T
    print(axes.shape)
else:
    __, axes = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey="row", squeeze=False)
for axis in axes.flatten():
    axis.grid(True)
    axis.axhline(0, color="black", linewidth=1)
    axis.axvline(0, color="black", linewidth=1)

data_max = max(abs(dat.x.imag).max() for dat in (gf_lay_w, gf_imp_w))
ax: plt.Axes = axes[-1, 0]
ax.set_ylim(bottom=-1.05*data_max, top=0.05*data_max)
ax = axes[0, 0]
ax.set_ylim(bottom=0.05*data_max, top=-1.05*data_max)

# gf_lay_w.x[0] *= -1  # check ifspins are correct

#
# plot data
#
for ax, gf_lay_ll, gf_lay_ll_err in zip(axes.reshape((-1)), gf_lay_w.x.reshape((-1, N_w)),
                                        gf_lay_w.err.reshape((-1, N_w))):
    plot.err_plot(x=omega.real, y=gf_lay_ll.imag, yerr=gf_lay_ll_err.imag, axis=ax,
                  fmt='--', label='Gf_lay')

for ax, gf_imp_ll, gf_imp_ll_err in zip(axes.reshape((-1)), gf_imp_w.x.reshape((-1, N_w)),
                                        gf_imp_w.err.reshape((-1, N_w))):
    plot.err_plot(x=omega.real, y=gf_imp_ll.imag, yerr=gf_imp_ll_err.imag, axis=ax,
                  fmt='--', label='Gf_imp')


spin_strs = ('↑', '↓')
for sp, ax in zip(spin_strs, axes[:, 0]):
    ax.set_ylabel(rf"$\Im G_{{{sp}l}}(i\omega_n)$")

for lay, ax in zip(imp_data.keys(), axes[0]):
    ax.set_title(f"l = {lay}")

for ax in axes[-1]:
    ax.set_xlabel("ω")

axes[-1, -1].legend()

plt.tight_layout()
plt.show()
