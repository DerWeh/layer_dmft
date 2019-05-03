"""Add temperature from 'init.py' to layer_output.

Temperature was added as necessity after *0.2.0-19-g7577bd9*.

"""
import numpy as np
from layer_dmft import layer_dmft, util, dataio

with util.local_import():
    from init import prm

if __name__ == '__main__':
    it, last_data = dataio.get_last_iter(dataio.LAY_OUTPUT)
    with np.load(last_data) as dat:
        saved = dict(dat)
    saved['temperature'] = prm.T
    np.savez_compressed(last_data, **saved)
    print(f"Added temperature T={prm.T} from 'init.py' to {last_data}")
