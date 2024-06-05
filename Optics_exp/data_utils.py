import os

import numpy as np


def read_data(r0: float, exp_name: str = 'optics') -> (np.ndarray, np.ndarray):
    """
    Reads grid and Poynting vector data from files based on the fixed r0 value.

    Args:
        exp_name: The name of the experiment.
        r0 (float): The fixed r0 value used to locate the data files.

    Returns:
        tuple: A tuple containing the grid data and Poynting vector data.
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), f'{exp_name}_data/T(H) r0={r0}'))
    grid_file = os.path.join(data_dir, f'grid_{r0}.txt')
    poynting_vec_file = os.path.join(data_dir, f'T_av_{r0}.txt')
    grid = np.genfromtxt(grid_file, delimiter=',')
    poynting_vec = np.genfromtxt(poynting_vec_file, delimiter=',')
    return grid, poynting_vec
