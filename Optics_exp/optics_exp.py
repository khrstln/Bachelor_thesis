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
solver_img_dir = os.path.join(os.path.dirname(__file__), 'optics_intermediate')  # directory for solver charts


def split_data(m_grid: np.ndarray, poynting_vec: np.ndarray, test_size: float = 0.2,
               random_state: int = 0) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    poynting_vec_train, poynting_vec_test, grid_train, grid_test = train_test_split(poynting_vec, m_grid,
                                                                                    test_size=test_size,
                                                                                    random_state=random_state)
    inds_train = np.argsort(grid_train)
    inds_test = np.argsort(grid_test)
    m_grid_train = np.sort(grid_train)
    m_grid_test = np.sort(grid_test)
    poynting_vec_train = poynting_vec_train[inds_train]
    poynting_vec_test = poynting_vec_test[inds_test]
    return m_grid_train, m_grid_test, poynting_vec_train, poynting_vec_test


def get_result_dir(exp_name: str) -> str:
    results_dir = os.path.join(os.path.dirname(__file__), fr'results\results_{exp_name}')
    if not (os.path.isdir(results_dir)):
        os.mkdir(results_dir)
    return results_dir


def save_split_exp_data(r0: int | float, m_grid_train: np.ndarray,
                        m_grid_test: np.ndarray, poynting_vec_train: np.ndarray,
                        poynting_vec_test: np.ndarray, results_dir: str) -> None:
    split_exp_data_dir = os.path.join(results_dir, 'split exp data')
    if not (os.path.isdir(split_exp_data_dir)):
        os.mkdir(split_exp_data_dir)

    grid_train_filename = os.path.join(split_exp_data_dir, fr'grid_train_{r0}.txt')
    grid_test_filename = os.path.join(split_exp_data_dir, fr'grid_test_{r0}.txt')
    poynting_vec_train_filename = os.path.join(split_exp_data_dir, fr'poynting_vec_train_{r0}.txt')
    poynting_vec_test_filename = os.path.join(split_exp_data_dir, fr'poynting_vec_test_{r0}.txt')

    np.savetxt(grid_train_filename, m_grid_train)
    np.savetxt(grid_test_filename, m_grid_test)
    np.savetxt(poynting_vec_train_filename, poynting_vec_train)
    np.savetxt(poynting_vec_test_filename, poynting_vec_test)


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
        x2 = 7
        axins = ax.inset_axes((20.0, -0.6, 35.0, 0.4),
                              xlim=(x1, x2), ylim=(y1, y2), transform=ax.transData, xticklabels=[],
                              yticklabels=[])
        axins.set_xticks(np.arange(0, 7.01, 1))
    else:
        x2 = 15
        axins = ax.inset_axes((20.0, -0.6, 35.0, 0.4),
                              xlim=(x1, x2), ylim=(y1, y2), transform=ax.transData, xticklabels=[],
                              yticklabels=[])
        axins.set_xticks(np.arange(0, 15.01, 2))
    return axins


def set_inserted_ax(axins: Axes, m_grid_train: np.ndarray, m_grid_test: np.ndarray,
                    poynting_vec_train: np.ndarray, poynting_vec_test: Any, pred_solution_train: Any,
                    start_y: int | float = -1, stop_y: int | float = 0.1, step_y: int | float = 0.2,
                    add_train_data: bool = False) -> None:
    axins.set_yticks(np.arange(start_y, stop_y, step_y))
    axins.grid(True)
    if add_train_data:
        axins.plot((m_grid_train / wave_length), poynting_vec_train, '+', label='Training data')
    axins.plot((m_grid_test / wave_length), poynting_vec_test, '.', color='black')
    axins.plot((m_grid_train / wave_length), pred_solution_train.detach().numpy(), color='r')


def set_main_ax(ax: Axes, axins: Axes, m_grid_train: np.ndarray, m_grid_test: np.ndarray,
                poynting_vec_train: np.ndarray, poynting_vec_test: np.ndarray, pred_solution_train: Any,
                rmse: float, mape: float, add_train_data: bool = False) -> None:
    ax.indicate_inset_zoom(axins, edgecolor="black")
    ax.set_xticks(np.arange(0, np.max((m_grid_test / wave_length)) + 10, 10))
    ax.set_yticks(np.arange(-1., 0.5, 0.2))
    if add_train_data:
        ax.plot((m_grid_train / wave_length), poynting_vec_train, '+', label='Training data')
    ax.plot((m_grid_test / wave_length), poynting_vec_test, '.', color='black', label='Test data')
    ax.plot((m_grid_train / wave_length), pred_solution_train.detach().numpy(), color='r',
            label='Solution of the discovered DE')
    ax.text(70, -0.8,
            f'RMSE = {rmse:.2e}', fontsize=10)
    # ax.text(70, -0.75,
    #         f'MAPE = {mape:.2e}', fontsize=10)
    ax.grid(True)


def set_plot(r0: int | float, img_filename: str, save_solutions: bool = True, add_legend: bool = False) -> None:
    if add_legend:
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
                  m_grid_test: np.ndarray, poynting_vec_train: np.ndarray, poynting_vec_test: np.ndarray,
                  pred_solution_train: Any, pred_solution_test: Any, results_dir: str,
                  save_solutions: bool = True, add_legend: bool = False) -> None:
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot()

    axins = get_inserted_ax(ax, r0)
    set_inserted_ax(axins, m_grid_train, m_grid_test, poynting_vec_train, poynting_vec_test, pred_solution_train)

    rmse = np.sqrt(mean_squared_error(poynting_vec_test, pred_solution_test.detach().numpy()))
    mape = mean_absolute_percentage_error(poynting_vec_test, pred_solution_test.detach().numpy())
    set_main_ax(ax, axins, m_grid_train, m_grid_test, poynting_vec_train, poynting_vec_test,
                pred_solution_train, rmse, mape)
    sln_img_dir = os.path.join(results_dir, 'solutions visualization')
    if not (os.path.isdir(sln_img_dir)):
        os.mkdir(sln_img_dir)
    img_filename = os.path.join(sln_img_dir, fr'sln_{r0}_{i}_{run}.png')
    set_plot(r0, img_filename, save_solutions=save_solutions, add_legend=add_legend)


def save_solution_data(r0: int | float, i: int, run: int, pred_solution: Any,
                       results_dir: str, training: bool = True) -> None:
    sln_data_dir = os.path.join(results_dir, 'solutions data')
    if not (os.path.isdir(sln_data_dir)):
        os.mkdir(sln_data_dir)
    sln_data_filename = os.path.join(sln_data_dir, fr'sln_data_training_{r0}_{i}_{run}.txt') if training \
        else os.path.join(sln_data_dir, fr'sln_data_test_{r0}_{i}_{run}.txt')
    np.savetxt(sln_data_filename, pred_solution.detach().numpy())


def save_txt_for_equations(r0: int | float, i: int, run: int,
                           results_dir: str, text_eq: str) -> None:
    txt_dir = os.path.join(results_dir, 'text equations')
    if not (os.path.isdir(txt_dir)):
        os.mkdir(txt_dir)
    txt_filename = os.path.join(txt_dir, fr'eqn_{r0}_{i}_{run}.txt')
    with open(txt_filename, 'w') as the_file:
        the_file.write(text_eq)


def optics_exp(r0: int | float, exp_name: str = 'optics', nruns: int = 1, solve_equations: bool = True,
               factors_max_number: int = 1, poly_order: int = 4, variable_names=None,
               max_deriv_order: int | tuple = (2,), equation_terms_max_number: int = 5,
               data_fun_pow: int = 1, training_epde_epochs: int = 100, training_tedeous_epochs: int = 10000) -> None:
    """
    Runs an optics experiment with the specified parameters.

    Args:
        r0 (int | float): The radius value in micrometers.
        exp_name (str): The name of the experiment (default is 'optics').
        nruns (int): The number of runs of the epde_discovery (default is 1).
        solve_equations (bool): Flag to solve equations (default is True).
        factors_max_number (int): Maximum number of factors in a term (default is 1).
        poly_order (int): Order of family of tokens for polynomials (default is 4).
        variable_names (list[str]): List of variable names (default is ['I']).
        max_deriv_order (int | tuple): Maximum derivative order (default is (2,)).
        equation_terms_max_number (int): Maximum number of equation terms (default is 5).
        data_fun_pow (int): The highest power of derivative-like token in the equation (default is 1).
        training_epde_epochs (int): Number of training epochs for EPDE (default is 100).
        training_tedeous_epochs (int): Number of training epochs for TEDEouS (default is 10000).

    Returns:
        None
    """

    if variable_names is None:
        variable_names = ['I']
    m_grid, poynting_vec = read_data(r0)
    m_grid_train, m_grid_test, poynting_vec_train, poynting_vec_test = split_data(m_grid, poynting_vec)
    results_dir = get_result_dir(exp_name)
    save_split_exp_data(r0, m_grid_train, m_grid_test, poynting_vec_train, poynting_vec_test, results_dir)
    for run in range(nruns):
        epde_search_obj = epde_discovery((m_grid_train / wave_length), poynting_vec_train,
                                         factors_max_number=factors_max_number, poly_order=poly_order,
                                         variable_names=variable_names, max_deriv_order=max_deriv_order,
                                         equation_terms_max_number=equation_terms_max_number,
                                         data_fun_pow=data_fun_pow, training_epochs=training_epde_epochs)

        text_eq = epde_search_obj.equations(only_print=False, only_str=True, num=1)[0]
        eqs_solver_form = get_equations_solver_form(epde_search_obj)

        for i, eq in enumerate(eqs_solver_form):
            if solve_equations:
                pred_solution_train, pred_solution_test = solver_solution(eq[0][1], poynting_vec,
                                                                          (m_grid_train / wave_length),
                                                                          (m_grid_test / wave_length), solver_img_dir,
                                                                          training_epochs=training_tedeous_epochs)
                draw_solution(r0, i, run, m_grid_train, m_grid_test, poynting_vec_train, poynting_vec_test,
                              pred_solution_train, pred_solution_test, results_dir, save_solutions=True,
                              add_legend=False)
                save_solution_data(r0, i, run, pred_solution_train, results_dir)  # сохраняю данные, которые выдает
                # численная модель решения на тренировочном наборе, вроде для рисования графиков как раз результаты
                # на тренировочном наборе нужны, а вот ошибка измеряется на тестовом
                save_solution_data(r0, i, run, pred_solution_test, results_dir, training=False)
            save_txt_for_equations(r0=r0, i=i, run=run, results_dir=results_dir, text_eq=text_eq[i])
