import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             mean_absolute_percentage_error)
from matplotlib.axes import Axes
from typing import Any
from epde.interface.interface import EpdeSearch

from discovery_utils import epde_discovery
from solver_utils import solver_solution
from data_utils import read_data

sys.path.append('../')
sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
mpl.rcParams.update(mpl.rcParamsDefault)

wave_length = 0.5  # wavelength in micrometers
img_dir = os.path.join(os.path.dirname(__file__), 'optics_intermediate')  # directory for solver charts


def split_data(m_grid: np.ndarray, poynting_vec: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    poynting_vec_train, poynting_vec_test, grid_train, grid_test = train_test_split(poynting_vec, m_grid, test_size=0.2,
                                                                                    random_state=0)
    inds_train = np.argsort(grid_train)
    inds_test = np.argsort(grid_test)
    m_grid_train = np.sort(grid_train)
    m_grid_test = np.sort(grid_test)
    poynting_vec_train = poynting_vec_train[inds_train]
    poynting_vec_test = poynting_vec_test[inds_test]
    return m_grid_train, m_grid_test, poynting_vec_train, poynting_vec_test


def get_result_dir(exp_name: str) -> str:
    results_dir = os.path.join(os.path.dirname(__file__), f'results_{exp_name}')
    if not (os.path.isdir(results_dir)):
        os.mkdir(results_dir)
    return results_dir


def get_equations_solver_form(epde_search_obj: EpdeSearch) -> list:
    return epde_search_obj.solver_forms()[0]


def get_inserted_ax(ax: Axes, r0: int | float) -> Axes:
    x1, x2, y1, y2 = 0.01, 3, -1.05, 0.05
    if r0 < 0.3:
        axins = ax.inset_axes((20.0, -0.6, 35.0, 0.4),
                              xlim=(x1, x2), ylim=(y1, y2), transform=ax.transData, xticklabels=[],
                              yticklabels=[])
        axins.set_xticks(np.arange(0, 3.01, 1))
    elif r0 < 0.5:
        y1 = 7
        axins = ax.inset_axes((20.0, -0.6, 35.0, 0.4),
                              xlim=(x1, x2), ylim=(y1, y2), transform=ax.transData, xticklabels=[],
                              yticklabels=[])
        axins.set_xticks(np.arange(0, 7.01, 1))
    else:
        y1 = 15
        axins = ax.inset_axes((20.0, -0.6, 35.0, 0.4),
                              xlim=(x1, x2), ylim=(y1, y2), transform=ax.transData, xticklabels=[],
                              yticklabels=[])
        axins.set_xticks(np.arange(0, 15.01, 2))
    return axins


def set_inserted_ax(axins: Axes, m_grid_test: np.ndarray, m_grid_train: np.ndarray,
                    poynting_vec_test: Any, pred_solution_train: Any,
                    start_y: int | float = -1, stop_y: int | float = 0.1, step_y: int | float = 0.2) -> None:
    axins.set_yticks(np.arange(start_y, stop_y, step_y))
    axins.grid(True)
    axins.plot((m_grid_test / wave_length), poynting_vec_test, '+')
    axins.plot((m_grid_train / wave_length), pred_solution_train.detach().numpy(), color='r')


def set_main_ax(ax: Axes, axins: Axes, m_grid_train: np.ndarray, m_grid_test: np.ndarray,
                poynting_vec_test: np.ndarray, pred_solution_train: Any, rmse: float, mape: float) -> None:
    ax.indicate_inset_zoom(axins, edgecolor="black")
    ax.set_xticks(np.arange(0, np.max((m_grid_test / wave_length)) + 10, 10))
    ax.set_yticks(np.arange(-1., 0.5, 0.2))
    ax.plot((m_grid_test / wave_length), poynting_vec_test, '+', label='Exp data')
    ax.plot((m_grid_train / wave_length), pred_solution_train.detach().numpy(), color='r',
            label='Solution of the discovered DE')
    ax.text(70, -0.9,
            f'RMSE = {rmse:.2e}', fontsize=10)
    ax.text(70, -0.8,
            f'MAPE = {mape:.2e}', fontsize=10)
    ax.grid(True)


def set_plot(r0: int | float, img_filename: str, save_solutions: bool = True) -> None:
    plt.legend(loc=4)
    plt.title(fr"$I_y(H), r_0 = {r0}\mu m$, $\lambda = {wave_length} \mu m$")
    plt.xlabel(r'$H / \lambda$')
    plt.ylabel(r'$I_y(H)$')
    if save_solutions:
        plt.savefig(img_filename)
        plt.close()
    else:
        plt.plot()


def draw_solution(r0: int | float, i: int, run: int, m_grid_train: np.ndarray,
                  m_grid_test: np.ndarray, poynting_vec_test: np.ndarray,
                  pred_solution_train: Any, pred_solution_test: Any, results_dir: str,
                  save_solutions: bool = True) -> None:
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot()

    axins = get_inserted_ax(ax, r0)
    set_inserted_ax(axins, m_grid_test, m_grid_train, poynting_vec_test, pred_solution_train)

    rmse = np.sqrt(mean_squared_error(poynting_vec_test, pred_solution_test.detach().numpy()))
    mape = mean_absolute_percentage_error(poynting_vec_test, pred_solution_test.detach().numpy())
    set_main_ax(ax, axins, m_grid_train, m_grid_test, poynting_vec_test, pred_solution_train, rmse, mape)

    img_filename = os.path.join(results_dir, fr'solutions visualization\sln_{r0}_{i}_{run}.png')
    set_plot(r0, img_filename, save_solutions=save_solutions)


def save_solution_data(r0: int | float, i: int, run: int, pred_solution: Any, results_dir: str) -> None:
    sln_data_filename = os.path.join(results_dir, fr'solutions data\sln_data_{r0}_{i}_{run}.txt')
    np.savetxt(sln_data_filename, pred_solution.detach.numpy())


def save_splitted_exp_data(r0: int | float, m_grid_train: np.ndarray,
                           m_grid_test: np.ndarray, poynting_vec_train: np.ndarray,
                           poynting_vec_test: np.ndarray, results_dir: str) -> None:
    grid_train_filename = os.path.join(results_dir, fr'splitted exp data\grid_train_{r0}.txt')
    grid_test_filename = os.path.join(results_dir, fr'splitted exp data\grid_test_{r0}.txt')
    poynting_vec_train_filename = os.path.join(results_dir, fr'splitted exp data\poynting_vec_train_{r0}.txt')
    poynting_vec_test_filename = os.path.join(results_dir, fr'splitted exp data\poynting_vec_test_{r0}.txt')
    np.savetxt(grid_train_filename, m_grid_train)
    np.savetxt(grid_test_filename, m_grid_test)
    np.savetxt(poynting_vec_train_filename, poynting_vec_train)
    np.savetxt(poynting_vec_test_filename, poynting_vec_test)


def optics_exp(r0: int | float, exp_name: str = 'optics', nruns: int = 1, solve_equations: bool = True) -> None:
    """
    Perform an optics experiment with EPDE discovery and solver solution for a given r0 value.

    Args:
        r0: The value for r0 in micrometers in the experiment.
        exp_name: The name of the experiment (default is 'optics').
        nruns: The number of runs for the experiment (default is 1).
        solve_equations: Flag to indicate whether to solve the discovered equations (default is True).

    Returns:
        None
    """
    m_grid, poynting_vec = read_data(r0, exp_name)
    m_grid_train, m_grid_test, poynting_vec_train, poynting_vec_test = split_data(m_grid, poynting_vec)
    results_dir = get_result_dir(exp_name)
    save_splitted_exp_data(r0, m_grid_train, m_grid_test, poynting_vec_train, poynting_vec_test, results_dir)
    for run in range(nruns):
        epde_search_obj = epde_discovery((m_grid_train / wave_length), poynting_vec_train, factors_max_number=1,
                                         poly_order=4, variable_names=['I'], max_deriv_order=(2,),
                                         equation_terms_max_number=5, data_fun_pow=1)

        text_eq = epde_search_obj.equations(only_print=False, only_str=True, num=1)[0]
        eqs_solver_form = get_equations_solver_form(epde_search_obj)

        for i, eq in enumerate(eqs_solver_form):
            if solve_equations:
                pred_solution_train, pred_solution_test = solver_solution(eq[0][1], poynting_vec,
                                                                          (m_grid_train / wave_length),
                                                                          (m_grid_test / wave_length), img_dir=img_dir)
                draw_solution(r0, i, run, m_grid_train, m_grid_test, poynting_vec_test,
                              pred_solution_train, pred_solution_test, results_dir, save_solutions=True)
                save_solution_data(r0, i, run, pred_solution_train, results_dir)  # сохраняю данные, которые выдает
                # численная модель решения на тренировочном наборе, вроде для рисования графиков как раз результаты
                # на тренировочном наборе нужны, а вот ошибка измеряется на тестовом
            txt_filename = os.path.join(results_dir, fr'text equations\eqn_{r0}_{i}_{run}.txt')
            with open(txt_filename, 'w') as the_file:
                the_file.write(text_eq[i])
