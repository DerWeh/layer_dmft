"""Helper to plot impurity data."""
from typing import Union
import numpy as np
import matplotlib.pyplot as plt

from layer_dmft import dataio


def plot(data, data_err=None, axes=None, **kwds):
    """Consistently plot data with and without errors."""
    default_args = {'linestyle': '--', 'marker': 'x'}
    if axes is None:
        axes = plt.gca()
    if data_err is None:
        axes.plot(data, **kwds, **default_args)
    else:
        axes.errorbar(np.arange(data.size), data, yerr=data_err,
                      **kwds, **default_args)


def plot_impurity_data(key: str, layer=0, it: Union[int, str] = -1):
    """Make a 'reasonable' plot of the quantity `key`.

    Parameters
    ----------
    key : str
        The name of the quantity.
    layer : int, optional
        The impurity model corresponding to `layer`.
    it : int or str, optional
        Which iteration number to plot (default: last). If `it` is an `int`,
        it will be used as index, thus it is a *relative* iteration number.
        `str` will be used as an *absolute* iteration number.

    """
    imp_obj = dataio.ImpurityData()
    try:
        iter_num = tuple(imp_obj.iterations)[it]
    except TypeError:  # maybe a string was used for an absolute number
        iter_num = int(it)
        if iter_num not in imp_obj.iterations:
            print(f"`{it!r}` no valid iteration. "
                  f"Available iterations:\t{tuple(imp_obj.iterations)}")
            raise SystemExit()
    imp_data = imp_obj.iter(iter_num)

    try:
        data = np.array([imp_lay[key] for imp_lay in imp_data.values()])
    except KeyError:
        print(f"'{key}' not in impurity data. "
              # f"available keys:\n{tuple(tuple(imp_data.values())[0].keys())}")
              "available keys:\n",
              *tuple(imp_data.values())[0].keys())
        raise SystemExit()
    try:
        data_err = np.array([imp_lay[key + '_err'] for imp_lay in imp_data.values()])
    except KeyError:
        data_err = np.zeros_like(data, dtype=bool)
    data = data[layer:layer+1] if layer is not None else data
    data_err = data_err[layer:layer+1] if layer is not None else data_err

    COMPLEX = True if np.any(np.iscomplex(data)) else False
    SPIN_DEP = True if data.ndim > 2 and data.shape[1] == 2 else False

    nrows = 2 if COMPLEX else 1
    __, axes = plt.subplots(nrows=nrows, sharex=True, squeeze=False,
                            num=key+f" iteration {iter_num}")
    axes = axes[:, 0]

    for dat_lay, dat_lay_err in zip(data, data_err):
        if SPIN_DEP:
            if COMPLEX:
                plot(dat_lay[0].imag, data_err=dat_lay_err[0], label='Im up', axes=axes[0])
                plot(dat_lay[1].imag, data_err=dat_lay_err[1], label='Im dn', axes=axes[0])
                plot(dat_lay[0].real, data_err=dat_lay_err[0], label='Re up', axes=axes[1])
                plot(dat_lay[1].real, data_err=dat_lay_err[1], label='Re dn', axes=axes[1])
            else:
                plot(dat_lay[0], data_err=dat_lay_err[0], label='up')
                plot(dat_lay[1], data_err=dat_lay_err[1], label='dn')
        else:
            if COMPLEX:
                plot(dat_lay.imag, data_err=dat_lay_err, axes=axes[0])
                plot(dat_lay.real, data_err=dat_lay_err, axes=axes[1])
            else:
                plot(dat_lay, data_err=dat_lay_err)
                plot(dat_lay, data_err=dat_lay_err)

    for ax in axes:
        ax.legend()

    if key.endswith('_iw'):
        plt.xlabel('n')
    if key.endswith('_tau'):
        plt.xlabel('tau')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import fire
    fire.Fire(plot_impurity_data)
