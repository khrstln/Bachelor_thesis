from pathlib import Path
import numpy as np


def get_data(r0: int | float) -> (np.ndarray, np.ndarray):
    """
    Reads and gets grid and Poynting vector data from files based on the fixed r0 value.

    Args:
        r0 (float): The fixed r0 value used to locate the data files. This is the radius of the dielectric
        inclusions in 2D supercell model of the inhomogeneous layer.

    Returns:
        tuple: A tuple containing the grid data and Poynting vector data.
    """
    data_dir = Path.cwd() / 'data' / 'new optics_data' / f'T(H) r0={r0}'
    grid_file = data_dir / f'grid_{r0}.txt'
    poynting_vec_file = data_dir / f'T_av_{r0}.txt'
    grid = np.genfromtxt(grid_file, delimiter=',')
    poynting_vec = np.genfromtxt(poynting_vec_file, delimiter=',')
    return grid, poynting_vec
