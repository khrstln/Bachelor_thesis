from pathlib import Path
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, mean_absolute_percentage_error)
from matplotlib.axes import Axes

from discovery_tools import epde_discovery
from solver_tools import get_solution
from data_tools import get_data
from results_analysis_tools import get_results_dir

mpl.rcParams.update(mpl.rcParamsDefault)

solver_img_dir = str(Path.cwd() / 'optics_intermediate')  # directory for solver charts


def get_split_data(r0: int | float, test_size: float = 0.2,
                   random_state: int = 0) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    m_grid, poynting_vec = get_data(r0)
    poynting_vec_training, poynting_vec_test, grid_training, grid_test = train_test_split(poynting_vec, m_grid,
                                                                                          test_size=test_size,
                                                                                          random_state=random_state)
    inds_training = np.argsort(grid_training)
    inds_test = np.argsort(grid_test)
    m_grid_training = np.sort(grid_training)
    m_grid_test = np.sort(grid_test)
    poynting_vec_training = poynting_vec_training[inds_training]
    poynting_vec_test = poynting_vec_test[inds_test]
    return m_grid_training, m_grid_test, poynting_vec_training, poynting_vec_test


def save_split_exp_data(r0: int | float, m_grid_training: np.ndarray,
                        m_grid_test: np.ndarray, poynting_vec_training: np.ndarray,
                        poynting_vec_test: np.ndarray, results_dir: Path) -> None:
    split_exp_data_dir = results_dir / 'split exp data'
    if not (split_exp_data_dir.exists()):
        split_exp_data_dir.mkdir()

    grid_training_filename = split_exp_data_dir / fr'grid_training_{r0}.txt'
    grid_test_filename = split_exp_data_dir / fr'grid_test_{r0}.txt'
    poynting_vec_training_filename = split_exp_data_dir / fr'poynting_vec_training_{r0}.txt'
    poynting_vec_test_filename = split_exp_data_dir / fr'poynting_vec_test_{r0}.txt'

    np.savetxt(grid_training_filename, m_grid_training)
    np.savetxt(grid_test_filename, m_grid_test)
    np.savetxt(poynting_vec_training_filename, poynting_vec_training)
    np.savetxt(poynting_vec_test_filename, poynting_vec_test)


def get_eqs_solver_text_form(grid_training: np.ndarray, poynting_vec_training: np.ndarray, pop_size: int,
                             factors_max_number: int, poly_order: int, training_epde_epochs: int,
                             variable_names: [str], max_deriv_order: (int,),
                             equation_terms_max_number: int, data_fun_pow: int) -> (list, list):
    epde_search_obj = epde_discovery(grid_training, poynting_vec_training, pop_size=pop_size,
                                     factors_max_number=factors_max_number, poly_order=poly_order,
                                     training_epochs=training_epde_epochs, variable_names=variable_names,
                                     max_deriv_order=max_deriv_order,
                                     equation_terms_max_number=equation_terms_max_number,
                                     data_fun_pow=data_fun_pow)
    return epde_search_obj.solver_forms()[0], epde_search_obj.equations(only_print=False, only_str=True, num=1)[0]


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


def set_inserted_ax(axins: Axes, grid_training: np.ndarray, grid_test: np.ndarray,
                    poynting_vec_training: np.ndarray, poynting_vec_test: np.ndarray,
                    pred_solution_training: torch.Tensor, add_training_data: bool = False,
                    start_y: int | float = -1, stop_y: int | float = 0.1, step_y: int | float = 0.2) -> None:
    axins.set_yticks(np.arange(start_y, stop_y, step_y))
    axins.grid(True)
    if add_training_data:
        axins.plot(grid_training, poynting_vec_training, '+', label='Training data')
    axins.plot(grid_test, poynting_vec_test, '.', color='black')
    axins.plot(grid_training, pred_solution_training.detach().numpy(), color='r')


def set_main_ax(ax: Axes, axins: Axes, grid_training: np.ndarray, grid_test: np.ndarray,
                poynting_vec_training: np.ndarray, poynting_vec_test: np.ndarray, pred_solution_training: torch.Tensor,
                rmse: float, add_training_data: bool = False) -> None:
    ax.indicate_inset_zoom(axins, edgecolor="black")
    ax.set_xticks(np.arange(0, np.max(grid_test) + 10, 10))
    ax.set_yticks(np.arange(-1., 0.5, 0.2))
    if add_training_data:
        ax.plot(grid_training, poynting_vec_training, '+', label='Training data')
    ax.plot(grid_test, poynting_vec_test, '.', color='black', label='Test data')
    ax.plot(grid_training, pred_solution_training.detach().numpy(), color='r',
            label='Solution of the discovered DE')
    ax.text(70, -0.8,
            f'RMSE = {rmse:.2e}', fontsize=10)

    ax.grid(True)


def set_plot(r0: int | float, wave_length: int | float, img_filename: Path, save_solutions: bool = True,
             add_legend: bool = False) -> None:
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


def draw_solution(r0: int | float, wave_length: int | float, i: int, run: int, m_grid_training: np.ndarray,
                  m_grid_test: np.ndarray, poynting_vec_training: np.ndarray, poynting_vec_test: np.ndarray,
                  pred_solution_training: torch.Tensor, pred_solution_test: torch.Tensor, results_dir: Path,
                  save_solutions: bool = True, add_legend: bool = False, add_training_data: bool = False) -> None:
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot()

    axins = get_inserted_ax(ax, r0)
    set_inserted_ax(axins, m_grid_training, m_grid_test, poynting_vec_training, poynting_vec_test,
                    pred_solution_training)

    rmse = np.sqrt(mean_squared_error(poynting_vec_test, pred_solution_test.detach().numpy()))
    set_main_ax(ax, axins, m_grid_training, m_grid_test, poynting_vec_training, poynting_vec_test,
                pred_solution_training, rmse, add_training_data)
    sln_img_dir = results_dir / 'solutions visualization'
    if not (Path(sln_img_dir).exists()):
        Path(sln_img_dir).mkdir()
    img_filename = sln_img_dir / fr'sln_{r0}_{i}_{run}.png'
    set_plot(r0, wave_length, img_filename, save_solutions=save_solutions, add_legend=add_legend)


def save_solution_data(r0: int | float, i: int, run: int, pred_solution: torch.Tensor,
                       results_dir: Path, training: bool = True) -> None:
    sln_data_dir = results_dir / 'solutions data'
    if not (sln_data_dir.exists()):
        sln_data_dir.mkdir()
    sln_data_filename = sln_data_dir / fr'sln_data_training_{r0}_{i}_{run}.txt' if training \
        else sln_data_dir / fr'sln_data_test_{r0}_{i}_{run}.txt'
    np.savetxt(sln_data_filename, pred_solution.detach().numpy())


def save_txt_form_equations(r0: int | float, i: int, run: int,
                            results_dir: Path, text_eq: str) -> None:
    txt_dir = results_dir / 'text equations'
    if not (txt_dir.exists()):
        txt_dir.mkdir()
    txt_filename = txt_dir / fr'eqn_{r0}_{i}_{run}.txt'
    with txt_filename.open(mode='w') as the_file:
        the_file.write(text_eq)


def start_solver(equation: [list], grid_training: np.ndarray, grid_test: np.ndarray,
                 poynting_vec_training: np.ndarray, poynting_vec_test: np.ndarray,
                 training_tedeous_epochs: int, results_dir: Path, r0: int | float,
                 wave_length: int | float, i: int, run: int) -> None:
    """
    Starts the solver process for solving the provided equation from the resulting population and
    saves the results.

    Args:
        equation: The equation to solve.
        grid_training: Training grid data.
        grid_test: Test grid data.
        poynting_vec_training: Training Poynting vector data.
        poynting_vec_test: Test Poynting vector data.
        training_tedeous_epochs: Number of training TEDEouS epochs.
        results_dir: Directory to save results.
        r0: The radius of the dielectric inclusions in 2D supercell model of the inhomogeneous layer
        value in micrometers.
        wave_length: The wavelength of the incident wave.
        i: Index value of the equation in the resulting population.
        run: The run number for this value of r0.

    Returns:
        None
    """

    pred_solution_training, pred_solution_test = get_solution(equation[0][1], poynting_vec_training,
                                                              grid_training,
                                                              grid_test, solver_img_dir,
                                                              training_epochs=training_tedeous_epochs)
    draw_solution(r0, wave_length, i, run, grid_training, grid_test, poynting_vec_training, poynting_vec_test,
                  pred_solution_training, pred_solution_test, results_dir, save_solutions=True,
                  add_legend=False)
    save_solution_data(r0, i, run, pred_solution_training, results_dir)
    save_solution_data(r0, i, run, pred_solution_test, results_dir, training=False)


def start_run(r0: int | float, wave_length: int | float, run: int, grid_training: np.ndarray,
              grid_test: np.ndarray, poynting_vec_training: np.ndarray,
              poynting_vec_test: np.ndarray, pop_size: int, factors_max_number: int,
              poly_order: int, training_epde_epochs: int, variable_names: [str],
              max_deriv_order: (int,), equation_terms_max_number: int, data_fun_pow: int,
              solve_equations: bool, training_tedeous_epochs: int, results_dir: Path) -> None:
    """
    Starts a run for solving equations from one population and saving results based on the provided parameters.

    Args:
        r0: The radius of the dielectric inclusions in 2D supercell model of the inhomogeneous layer
        value in micrometers.
        wave_length: The wavelength of the incident wave.
        run: The run number for this value of r0.
        grid_training: Training grid data.
        grid_test: Test grid data.
        poynting_vec_training: Training Poynting vector data.
        poynting_vec_test: Test Poynting vector data.
        pop_size: Population size for EPDE.
        factors_max_number: Maximum number of factors in a term.
        poly_order: Order of family of tokens for polynomials.
        training_epde_epochs: Number of training EPDE epochs.
        variable_names: Names of variables in a differential equation.
        max_deriv_order: Maximum derivative order in a differential equation.
        equation_terms_max_number: Maximum number of equation terms.
        data_fun_pow: The highest power of derivative-like token in the equation.
        solve_equations: Boolean indicating whether to solve equations.
        training_tedeous_epochs: Number of training TEDEouS epochs.
        results_dir: Directory to save results.

    Returns:
        None
    """

    eqs_solver_form, eqs_text_form = get_eqs_solver_text_form(grid_training=grid_training,
                                                              poynting_vec_training=poynting_vec_training,
                                                              pop_size=pop_size, factors_max_number=factors_max_number,
                                                              poly_order=poly_order,
                                                              training_epde_epochs=training_epde_epochs,
                                                              variable_names=variable_names,
                                                              max_deriv_order=max_deriv_order,
                                                              equation_terms_max_number=equation_terms_max_number,
                                                              data_fun_pow=data_fun_pow)
    for i, eq in enumerate(eqs_solver_form):
        if solve_equations:
            start_solver(eq, grid_training, grid_test, poynting_vec_training, poynting_vec_test,
                         training_tedeous_epochs, results_dir, r0, wave_length, i, run)
        save_txt_form_equations(r0=r0, i=i, run=run, results_dir=results_dir, text_eq=eqs_text_form[i])

