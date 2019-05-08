"""Helper to plot impurity data."""
import numpy as np
import matplotlib.pyplot as plt

from layer_dmft import dataio

it = None
key = 'self_energy_iw'
layer = 0

imp_obj = dataio.ImpurityData()

it = it if it else max(imp_obj.iterations)

imp_data = imp_obj.iter(it)

try:
    data = np.array([imp_lay[key] for imp_lay in imp_data.values()])
except KeyError:
    # FIXME
    print(f"'{key}' not in impurity data."
          f"available keys:\n{tuple(imp_data.values())[0].keys()}")
    raise SystemExit()
data = data[layer:layer+1] if layer is not None else data

COMPLEX = True if np.any(np.iscomplex(data)) else False
SPIN_DEP = True if data.ndim > 2 and data.shape[1] == 2 else False

nrows = 2 if COMPLEX else 1
__, axes = plt.subplots(nrows=nrows, sharex=True, squeeze=False)
print(axes.shape)
axes = axes[:, 0]



for dat_lay in data:
    if SPIN_DEP:
        if COMPLEX:
            axes[0].plot(dat_lay[0].imag, '--x', label='Im up')
            axes[0].plot(dat_lay[1].imag,  '--x',label='Im dn')
            axes[1].plot(dat_lay[0].real, '--x', label='Re up')
            axes[1].plot(dat_lay[1].real, '--x', label='Re dn')
        else:
            plt.plot(dat_lay[0], '--x', label='up')
            plt.plot(dat_lay[1], '--x', label='dn')
    else:
        if COMPLEX:
            axes[0].plot(dat_lay.imag)
            axes[1].plot(dat_lay.real, '--')
        else:
            plt.plot(dat_lay, '--x')
            plt.plot(dat_lay, '--x')


for ax in axes:
    ax.legend()
plt.show()
