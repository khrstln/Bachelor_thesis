import numpy as np
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt
from data_utils import read_data
import os
import sys
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

SMALL_SIZE = 12
matplotlib.rc('font', size=SMALL_SIZE)
matplotlib.rc('axes', titlesize=SMALL_SIZE)

script_dir = os.path.dirname(__file__)
if not os.path.exists(script_dir + r"\visula_data"):
    os.makedirs(script_dir + r"\visula_data")

r0_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
wave_length = 0.5
def func_to_approx(x, a, b, c):
    return a * np.exp(- c * x) + b

for i, r0_fix in enumerate(r0_arr):
    grid, I = read_data(r0_fix)
    grid = grid / wave_length
    popt, pcov = curve_fit(func_to_approx, grid, I)
    mse = mean_squared_error(I, func_to_approx(grid, *popt))
    fig = plt.figure(figsize=(16, 6), dpi=200)
    for j in [1, 2]:
        ax = fig.add_subplot(1, 2, j)
        if j == 2:
            if r0_fix < 0.3:
                ax.set_xticks(np.arange(0, 3, 1))
                ax.set_yticks(np.arange(-1., 0.1, 0.2))
                ax.set_xlim([-0.3, 3])
                ax.set_ylim([-1.05, 0.05])
            elif r0_fix < 0.5:
                ax.set_xticks(np.arange(0, 7, 1))
                ax.set_yticks(np.arange(-1., 0.1, 0.2))
                ax.set_xlim([-1, 7])
                ax.set_ylim([-1.05, 0.05])
            else:
                ax.set_xticks(np.arange(0, 15, 2))
                ax.set_yticks(np.arange(-1., 0.1, 0.2))
                ax.set_xlim([-1, 15])
                ax.set_ylim([-1.05, 0.05])
            ax.grid(True)
            ax.plot(grid, I, "+")
            ax.plot(grid, func_to_approx(grid, *popt), "r")
        else:
            ax.set_xticks(np.arange(0, np.max(grid), 10))
            ax.set_yticks(np.arange(-1., 0.1, 0.2))
            ax.text((np.max(grid) + np.min(grid)) / 2, -0.55, f'MSE = {mse:.2e}', fontsize=10)
            ax.grid(True)
            ax.plot(grid, I, "+", label="Exp data")
            ax.plot(grid, func_to_approx(grid, *popt), "r", \
                     label=fr"$Fit: {popt[0]:.3f} \cdot exp({np.round(popt[1], 3)}y) + {np.round(popt[2], 3)}$")
            plt.legend()
        plt.title(fr"$I_y(H), r_0 = {r0_fix}\mu m$, $\lambda = {wave_length} \mu m$")
        plt.xlabel(r'$H / \lambda$')
        plt.ylabel(r'$I_y(H)$')
        plt.savefig(script_dir + fr"\visula_data\exponential approximation\I(H) r_0={r0_fix}.png")
# plt.show()

