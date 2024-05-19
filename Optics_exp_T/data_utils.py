import os
import numpy as np

def read_data(r0_fix):
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), f'optics_data_T/T(H) r0={r0_fix}'))
    grid_file = os.path.join(data_dir, 'grid_{}.txt'.format(r0_fix))
    T_file = os.path.join(data_dir, 'T_av_{}.txt'.format(r0_fix))
    grid = np.genfromtxt(grid_file, delimiter=',')
    T = np.genfromtxt(T_file, delimiter=',')
    return grid, T