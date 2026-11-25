"""Read binary PCD files used by localization inputs and outputs."""
from pathlib import Path

import numpy as np


def read_pcd(path):
    path = Path(path)
    with path.open('rb') as stream:
        header = {}
        while True:
            line = stream.readline()
            if not line:
                raise ValueError('Incomplete PCD header')
            line = line.decode().strip()
            if not line or line.startswith('#'):
                continue
            key, *values = line.split()
            header[key] = values
            if key == 'DATA':
                break

        if header['DATA'] != ['binary']:
            raise ValueError('Expected binary PCD')
        kinds = {'F': 'f', 'I': 'i', 'U': 'u'}
        dtype = np.dtype([
            (name, '<' + kinds[kind] + size)
            for name, kind, size in zip(header['FIELDS'], header['TYPE'], header['SIZE'])
        ])
        return np.frombuffer(stream.read(), dtype=dtype, count=int(header['POINTS'][0]))
