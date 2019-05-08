"""Preliminary ploting skript for spinflip."""
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import gftools as gt
from gftools import pade as gtpade

from layer_dmft import util, plot
from layer_dmft.interface import sb_qmc

with util.local_import():
    from init import prm

data_iv = np.loadtxt(sb_qmc.OUTPUT_DIR + "/00-chi_tr_omega-t.dat", unpack=True)

iv = gt.matsubara_frequencies_b(data_iv[0], beta=prm.beta)
spin_flip1_iv = data_iv[1] + 1j*data_iv[2]
spin_flip2_iv = data_iv[3] + 1j*data_iv[4]
assert np.all(data_iv[0] == np.arange(spin_flip1_iv.size))


omega = np.linspace(-4, 4, num=1000, dtype=np.complex) + iv[1]/10
valid_w = omega[omega.real <= 0]
kind_gf = gtpade.KindGf(20, 100)
pade_prm = {'kind': kind_gf, 'threshold': 1e-2, 'valid_z': valid_w}
pade_prm = {'kind': kind_gf, 'threshold': np.infty, 'valid_z': valid_w}
avg_no_neg_imag = partial(gtpade.avg_no_neg_imag, z_in=iv, **pade_prm)

spin_flip1_w = avg_no_neg_imag(omega, fct_z=spin_flip1_iv)
spin_flip2_w = avg_no_neg_imag(omega, fct_z=spin_flip2_iv)
spin_flip_w = avg_no_neg_imag(omega, fct_z=(spin_flip1_iv + spin_flip2_iv)/2.)

plot.err_plot(omega.real, spin_flip1_w.x.real, spin_flip1_w.err.real)
plot.err_plot(omega.real, spin_flip2_w.x.real, spin_flip2_w.err.real)
plot.err_plot(omega.real, spin_flip_w.x.real, spin_flip_w.err.real)
plt.ylabel(r"$\Re \chi^{+-}$")
plt.xlabel(r"$\omega$")
plt.show()


plot.err_plot(omega.real, spin_flip1_w.x.imag, spin_flip1_w.err.imag)
plot.err_plot(omega.real, spin_flip2_w.x.imag, spin_flip2_w.err.imag)
plot.err_plot(omega.real, spin_flip_w.x.imag, spin_flip_w.err.imag)
plt.ylabel(r"$\Im \chi^{+-}$")
plt.xlabel(r"$\omega$")
plt.show()


