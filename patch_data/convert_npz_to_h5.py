"""Script to convert old data from version < 0.5.0 to '.h5' files."""
import logging

from pathlib import Path

import numpy as np
import xarray as xr

from layer_dmft.util import Dimensions as Dim
from layer_dmft.dataio import _get_iter, save_dataset


def get_coords(spin, layer, matsubara) -> dict:
    assert spin in (1, 2), f"Invalid value for spin: {spin}"
    coords = {Dim.sp: ['up']} if spin == 1 else {'spin': ['up', 'dn']}
    coords[Dim.lay] = np.arange(layer)
    coords[Dim.iws] = np.arange(matsubara)
    return coords


def npz2h5(filename: str):
    """Convert old 'npz' files to new 'h5' format using `xarray`s.

    Parameters
    ----------
    filename : str
        Name of the file to convert.

    """
    infile: Path = Path(filename).absolute()
    assert infile.is_file()
    assert infile.suffix in ('.npz', '.npy')

    copy = infile.parent / (infile.stem + 'old' + infile.suffix)

    logging.debug('Loading file %s', str(infile))
    data = np.load(infile)

    print(*data.keys())

    coords = get_coords(*data['gf_iw'].shape)

    gf_iw = xr.DataArray(data['gf_iw'], name='G$_{lat}(iw_n)$',
                         dims=[Dim.sp, Dim.lay, Dim.iws], coords=coords)
    self_iw = xr.DataArray(data['self_iw'], name='Σ($iw_n$)',
                           dims=[Dim.sp, Dim.lay, Dim.iws], coords=coords)
    occ = xr.DataArray(data['occ'], name='occupation', dims=[Dim.sp, Dim.lay],
                       coords={Dim.sp: coords[Dim.sp], Dim.lay: coords[Dim.lay]})
    layer_data = xr.Dataset(
        {'gf_iw': gf_iw, 'self_iw': self_iw, 'occ': occ},  # , 'onsite-paramters': prm.params},
        attrs={key: data[key] for key in data.keys() if key not in ['gf_iw', 'self_iw', 'occ']}
    )

    layer_data.attrs[Dim.it] = _get_iter(infile)

    logging.debug("Safe new 'h5' file.")
    save_dataset(layer_data, dir_=infile.parent, name=infile.stem, use_date=False)

    logging.debug("Move old file %s to 'old.npz'.", infile.name)
    infile.rename(copy)
    logging.info("Converted %s to 'h5' file %s", infile.name, infile.stem + '.h5')


if __name__ == '__main__':
    try:
        from fire import Fire
    except ImportError:
        logging.info("'Fire' found, fall back to basic 'sys' argument handling.")
        import sys
        npz2h5(sys.argv[1])
    else:
        Fire(npz2h5)
