import os
import sys
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from discovery_utils import epde_discovery, solver_solution
from data_utils import read_data
from tedeous.device import solver_device
from sklearn.metrics import mean_squared_error
sys.path.append('../')
sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
mpl.rcParams.update(mpl.rcParamsDefault)

wave_length = 0.5 # wavelength in micrometers

def optics_exp(r0_fix, exp_name='optics', trend_remove=False, custom_derivs=False,
               derivs_params={'interp_mode': 'NN', 'diff_mode': 'FD', 'diffs_plot': False, 'save_derivs': False},
               nruns=1, solve_equations=True):
    for run in range(nruns):
        solver_device('cpu')
        grid, I_lambda = read_data(r0_fix)
        bonudary = 0
        m_grid = grid
        derivs = None
        epde_search_obj = epde_discovery((m_grid / wave_length), I_lambda, boundary=bonudary, derivs=derivs, use_ann=False)
        # epde_search_obj = epde_discovery(m_grid, I_lambda, boundary=bonudary, derivs=derivs, use_ann=False)
        # epde_search_obj.visualize_solutions()
        res = epde_search_obj.solver_forms()
        text_eq = epde_search_obj.equations(only_print=False, only_str=True, num=1)
        text_eq = text_eq[0]
        eqs = res[0]
        results_dir = os.path.join(os.path.dirname(__file__), 'results_{}'.format(exp_name))
        if not (os.path.isdir(results_dir)):
            os.mkdir(results_dir)
        for i, eq in enumerate(eqs):
            if solve_equations:
                pred_solution = solver_solution(eq[0][1], I_lambda, (m_grid / wave_length))
                # pred_solution = solver_solution(eq[0][1], I_lambda, m_grid)
                # fig = plt.figure(figsize=(16, 6), dpi=200)
                # for j in [1, 2]: # Раскомментить эту строчку и строчку выше, если нужен приближенный график
                fig = plt.figure(dpi=200)
                for j in [1]:
                    # ax = fig.add_subplot(1, 2, j) # Тоже расскомментить, если нужен приближенный график
                    ax = fig.add_subplot()
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
                        ax.plot((m_grid / wave_length), I_lambda, '+')
                        ax.plot((m_grid / wave_length), pred_solution.detach().numpy(), color='r')
                    else:
                        ax.set_xticks(np.arange(0, np.max((m_grid / wave_length)), np.max((m_grid / wave_length) / 10)))
                        ax.set_yticks(np.arange(-1., 0.5, 0.2))
                        ax.plot((m_grid / wave_length), I_lambda, '+', label='Exp data')
                        ax.plot((m_grid / wave_length), pred_solution.detach().numpy(), color='r',
                                 label='Solution of the discovered DE')
                        mse = mean_squared_error(I_lambda, pred_solution.detach().numpy())
                        # ax.text((np.max(m_grid / wave_length) + np.min(m_grid / wave_length)) / 2, -0.55, f'MSE = {mse:.2e}', fontsize=10)
                        ax.text(0.5, -0.55,
                                f'MSE = {mse:.2e}', fontsize=10)
                        ax.grid(True)
                        plt.legend()
                        plt.title(fr"$I_y(H), r_0 = {r0_fix}\mu m$, $\lambda = {wave_length} \mu m$")
                plt.xlabel(r'$H / \lambda$')
                plt.ylabel(r'$I_y(H)$')
                img_filename = os.path.join(results_dir, r'solutions visualization\sln_{}_{}_{}.png'.format(r0_fix, i, run))
                plt.savefig(img_filename)
                plt.close()
            txt_filename = os.path.join(results_dir, r'text equations\eqn_{}_{}_{}.txt'.format(r0_fix, i, run))
            with open(txt_filename, 'w') as the_file:
                the_file.write(text_eq[i])


if __name__ == '__main__':

    for r0_fix in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        optics_exp(r0_fix, exp_name='optics')

    # for r0_fix in [0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    #     optics_exp(r0_fix, exp_name='optics')

    # for r0_fix in [0.7]:
    #     optics_exp(r0_fix, exp_name='optics')

