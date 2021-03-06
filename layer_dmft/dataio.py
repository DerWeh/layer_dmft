"""Handle in- and output for the layer_dmft loop."""
import warnings

from typing import Dict, Optional, Tuple, Union
from pathlib import Path
from weakref import finalize
from datetime import date
from collections import defaultdict, OrderedDict

import numpy as np
import xarray as xr

from layer_dmft.util import Dimensions as Dim

LAY_OUTPUT = "layer_output"
IMP_OUTPUT = "imp_output"


def save_data(dir_='.', *, name, compress=True, use_date=True, **data):
    """Save numpy arrays in `data` to file `dir_/name`.

    Parameters
    ----------
    dir_ : str, Path
        Directory where the data is written.
    name : str
        Filename
    compress : bool, optional
        Whether the data is written to compressed file or uncompressed (default: True).
    use_date : bool, optional
        Whether the filename is perpended by the date (default: True).
    data
        The data written to the file.

    """
    dir_ = Path(dir_).expanduser()
    dir_.mkdir(exist_ok=True)
    save_method = np.savez_compressed if compress else np.savez
    name = date.today().isoformat() + '_' + name if use_date else name
    save_method(dir_/name, **data)


def save_dataset(data, dir_='.', *, name, use_date=True, **kwds):
    """Save the `xr.Dataset` `data` to `dir_/name`.

    If no ending is specified, '.h5' will be appended.

    Parameters
    ----------
    dir_ : str, Path
        Directory where the data is written.
    name : str
        Filename
    compress : bool, optional
        Whether the data is written to compressed file or uncompressed (default: True).
    use_date : bool, optional
        Whether the filename is perpended by the date (default: True).
    data
        The data written to the file.

    """
    dir_ = Path(dir_).expanduser()
    dir_.mkdir(exist_ok=True)
    name = date.today().isoformat() + '_' + name if use_date else name
    dir_name = dir_ / name
    if not dir_name.suffix:
        dir_name = dir_name.with_suffix('.h5')
    data.to_netcdf(dir_name, engine="h5netcdf", invalid_netcdf=True,
                   format='NETCDF4', **kwds)


def _get_iter(file_object) -> Optional[int]:
    r"""Return iteration `it` number of file with the name '\*_iter{it}(_*)?.ENDING'."""
    return _get_anystring(file_object, name='iter')


def _get_layer(file_object) -> Optional[int]:
    r"""Return iteration `it` number of file with the name '\*_lay{it}(_*)?.ENDING'."""
    return _get_anystring(file_object, name='lay')


def _get_anystring(file_object, name: str) -> Optional[int]:
    r"""Return iteration `it` number of file with the name '\*_{`name`}{it}(_*)?.ENDING'."""
    basename = Path(file_object).stem
    ending = basename.split(f'_{name}')[-1]  # select part after '_iter'
    iter_num = ending.split('_')[0]  # drop everything after possible '_'
    try:
        it = int(iter_num)
    except ValueError:
        warnings.warn(f"Skipping unprocessable file: {file_object.name}")
        return None
    return it


def get_iter(dir_, num) -> Path:
    """Return the file of the output of iteration `num`."""
    iter_files = Path(dir_).glob(f'*_iter{num}*.*')

    paths = [iter_f for iter_f in iter_files if _get_iter(iter_f) == num]
    if not paths:
        raise AttributeError(f'Iterations {num} cannot be found.')
    if len(paths) > 1:
        raise AttributeError(f'Multiple occurrences of iteration {num}:\n'
                             + '\n'.join(str(element) for element in paths))
    return paths[0]


def get_last_iter(dir_) -> Tuple[int, Path]:
    """Return number and the file of the output of last iteration."""
    iter_files = Path(dir_).expanduser().glob('*_iter*.*')

    iters = {_get_iter(file_): file_ for file_ in iter_files}
    try:  # remove invalid item
        last_iter: int = max(iters.keys() - {None})  # type: ignore
    except ValueError:
        raise IOError(f"No valid iteration data available in {Path(dir_).expanduser()}")
    return last_iter, iters[last_iter]


def get_all_iter(dir_) -> Dict[int, Path]:
    """Return dictionary of files of the output with key `num`."""
    iter_files = Path(dir_).glob('*_iter*.*')
    path_dict = {_get_iter(iter_f): iter_f for iter_f in iter_files
                 if _get_iter(iter_f) is not None}
    return path_dict


def get_all_imp_iter(dir_) -> Dict[int, Dict]:
    """Return directory of {int(layer): output} with key `num`."""
    iter_files = Path(dir_).glob('*_iter*_lay*.*')
    path_dict: Dict[int, Dict] = defaultdict(dict)
    for iter_f in iter_files:
        it = _get_iter(iter_f)
        lay = _get_layer(iter_f)
        if (it is not None) and (lay is not None):
            path_dict[it][lay] = iter_f
    return path_dict


class LayerData:
    """Interface to saved layer data."""

    keys = {'gf_iw', 'self_iw', 'occ'}

    def __init__(self, dir_=LAY_OUTPUT):
        """Mmap all data from directory."""
        warnings.warn('Outdated methods, not certain to work anymore.', DeprecationWarning)
        if not Path(dir_).is_dir():
            raise NotADirectoryError(str(dir_))
        self._filname_dict = get_all_iter(dir_)
        self.mmap_dict = OrderedDict((key, self._autoclean_load(val, mmap_mode='r'))
                                     for key, val in sorted(self._filname_dict.items()))
        self.array = np.array(self.mmap_dict.values(), dtype=object)

    # FIXME: do it in a way, that doesn't delete referenced data if `LayerData` is deleted
    # e.g. wrap all data in data wrapper
    def _autoclean_load(self, *args, **kwds):
        data = np.load(*args, **kwds)

        def _test():
            data.close()
        finalize(self, _test)
        return data

    def iter(self, it: Union[str, int] = None, *, abs_it: int = None, rel_it: int = None,
             return_iternum=False):
        """Return data of iteration `it`."""
        if len(set((it, abs_it, rel_it)) - {None}) != 1:
            raise TypeError(f"{iter.__name__} takes exactly one argument.")
        if it is not None:
            try:
                abs_it = tuple(self.iterations)[it]
            except TypeError:
                abs_it = int(it)
        if rel_it is not None:
            abs_it = tuple(self.iterations)[rel_it]
        if abs_it not in self.iterations:
            raise TypeError(f"`{abs_it!r}` no valid iteration. "
                            f"Available iterations:\t{tuple(self.iterations)}")
        if return_iternum:
            return self.mmap_dict[abs_it], abs_it
        return self.mmap_dict[abs_it]

    @property
    def iterations(self):
        """Return list of iteration numbers."""
        return self.mmap_dict.keys()

    def __getitem__(self, key):
        """Emulate structured array behavior."""
        try:
            return self.mmap_dict[key]
        except KeyError:
            if key in self.keys:
                return np.array([data[key] for data in self.mmap_dict.values()])
            else:
                raise

    def __getattr__(self, item):
        """Access elements in `keys`."""
        if item in self.keys:
            return self[item]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")


def layer_data(dir_=None) -> xr.Dataset:
    """Load all data from `LAY_OUTPUT`."""
    if dir_ is None:
        dir_ = LAY_OUTPUT
    if not Path(dir_).is_dir():
        raise NotADirectoryError(str(dir_))
    files = OrderedDict(sorted(get_all_iter(dir_).items()))
    lay_dat = xr.open_mfdataset(files.values(), engine='h5netcdf', concat_dim=Dim.it,
                                combine='nested')
    lay_dat = lay_dat.assign_coords(**{Dim.it: list(files.keys())})
    return lay_dat


class ImpurityData:
    """Interface to saved impurity data."""

    keys = {'gf_iw', 'gf_tau', 'self_iw', 'self_tau'}

    def __init__(self, dir_=IMP_OUTPUT):
        """Mmap all data from directory."""
        if not Path(dir_).is_dir():
            raise NotADirectoryError(str(dir_))
        self._filname_dict = get_all_imp_iter(dir_)
        mmap_dict = OrderedDict()
        for iter_key, iter_dict in sorted(self._filname_dict.items()):
            mmap_dict[iter_key] = OrderedDict(
                (key, self._autoclean_load(val, mmap_mode='r'))
                for key, val in sorted(iter_dict.items())
            )
        self.mmap_dict = mmap_dict
        self.array = np.array(self.mmap_dict.values(), dtype=object)

    def _autoclean_load(self, *args, **kwds):
        data = np.load(*args, **kwds)

        def _test():
            data.close()
        finalize(self, _test)
        return data

    def iter(self, it: int):
        """Return data of iteration `it`."""
        return self.mmap_dict[it]

    @property
    def iterations(self):
        """Return list of iteration numbers."""
        return self.mmap_dict.keys()

    def __getitem__(self, key):
        """Emulate structured array behavior."""
        return self.mmap_dict[key]

    # def __getattr__(self, item):
    #     """Access elements in `keys`."""
    #     if item in self.keys:
    #         return self[item]
    #     raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")
