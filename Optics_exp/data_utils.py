import os

import numpy as np


def read_data(r0_fix):
    """
    Reads grid and Poynting vector data from files based on the fixed r0 value.

    Args:
        r0_fix (int): The fixed r0 value used to locate the data files.

    Returns:
        tuple: A tuple containing the grid data and Poynting vector data.
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), f'optics_data_T/T(H) r0={r0_fix}'))
    grid_file = os.path.join(data_dir, f'grid_{r0_fix}.txt')
    poynting_vec_file = os.path.join(data_dir, f'T_av_{r0_fix}.txt')
    grid = np.genfromtxt(grid_file, delimiter=',')
    poynting_vec = np.genfromtxt(poynting_vec_file, delimiter=',')
    return grid, poynting_vec
