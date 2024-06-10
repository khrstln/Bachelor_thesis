from experiment_tools import *
from results_analysis_tools import *
import pandas as pd
import itertools
import torch


def get_results_df(r0_list: [int | float], exp_name: str, pop_size: int,
                   nruns: int) -> pd.DataFrame:
    """
    Creates a pandas DataFrame containing results from equations for a list of initial values.

    Args:
        r0_list: List containing radius values.
        exp_name: Name of the experiment.
        pop_size: The population size.
        nruns: The number of runs.
    Returns:
        Pandas DataFrame with the results from the equations.
    """

    eqn_list = []
    for r0_fix in r0_list:
        eqn_list.extend(
            f'eqn_{r0_fix}_{i}_{j}.txt'
            for i, j in itertools.product(range(pop_size), range(nruns))
        )
    read_eq_dict = {
        eq: read_eqn(exp_name, eq)
        for eq in eqn_list
        if read_eqn(exp_name, eq) is not None
    }
    return pd.DataFrame(read_eq_dict).transpose().fillna(0)


def get_equation_latex_form(results_df: pd.DataFrame, eq_name: str) -> str:
    """
    Generates the LaTeX form of an equation based on the coefficients and terms from the results DataFrame.

    Args:
        results_df: Pandas DataFrame containing the results.
        eq_name: Name of the equation.

    Returns:
        LaTeX formatted string representing the equation.
    """

    rmse = results_df['rmse'][eq_name]
    results_df = results_df.drop('rmse', axis=1)
    coefs = np.array(results_df.loc[eq_name])
    terms = [fr'{coefs[i]:.3f} \cdot ' + results_df.columns[i].replace('C', '') for i in range(len(coefs)) if
             coefs[i] not in [0.0] and results_df.columns[i] != 'dI/dH']
    terms[0] = f"$${terms[0]}"
    eq_name = eq_name.replace('.txt', '')
    params = eq_name.split('_')
    params = {"$r_0$ = ": params[1], "index = ": params[2], "run = ": params[3], "rmse = ": f'{rmse:.3f}'}
    res = "$r_0$ = " + params["$r_0$ = "] + ", index = " + params["index = "] + ", run = " + params[
        "run = "] + ", rmse = " + params["rmse = "] + ": " + " + ".join(terms) + " = dI/dH$$" + "\n\n"
    res = res.replace('+ -', '- ')
    res = res.replace('dI/dH', r'\frac{dI}{dH}')
    res = res.replace(r'\cdot  ', '')
    return res


def save_total_results_csv(r0_list: [int | float], exp_name: str, pop_size: int,
                           nruns: int) -> None:
    """
    Saves the total results DataFrame to a CSV file for a given experiment.

    Args:
        r0_list: List containing radius values.
        exp_name: Name of the experiment.
        pop_size: The population size.
        nruns: The number of runs.

    Returns:
        None
    """

    results_df = get_results_df(r0_list, exp_name, pop_size, nruns)
    results_dir_name = get_results_dir(exp_name)
    results_df.to_csv(results_dir_name / fr'total_results_{exp_name}.csv')


def save_total_results_latex_form(r0_list: [int | float], exp_name: str, pop_size: int,
                                  nruns: int) -> None:
    """
    Saves the LaTeX form of total results equations to a Markdown file for a given experiment.

    Args:
        r0_list: List containing radius values.
        exp_name: Name of the experiment.
        pop_size: The population size.
        nruns: The number of runs.
    Returns:
        None
    """

    results_df = get_results_df(r0_list, exp_name, pop_size, nruns)

    results_dir_name = get_results_dir(exp_name)

    total_results_file_path = results_dir_name / f'total_results_{exp_name}.md'

    with total_results_file_path.open(mode='w') as equations_file:
        for eq_name in results_df.index:
            equation_str = get_equation_latex_form(results_df, eq_name)
            equations_file.write(equation_str)


def start_exp(r0: int | float, wave_length: int | float, exp_name: str = 'optics', nruns: int = 1,
              solve_equations: bool = True, pop_size: int = 5,
              factors_max_number: int = 1, poly_order: int = 4, variable_names=None,
              max_deriv_order: int | tuple = (2,), equation_terms_max_number: int = 5,
              data_fun_pow: int = 1, training_epde_epochs: int = 100, training_tedeous_epochs: int = 10000) -> None:
    """
    Runs an optics experiment with the specified parameters.

    Args:
        r0 (int | float): The radius of the dielectric inclusions in 2D supercell model of the inhomogeneous layer
        value in micrometers.
        wave_length: The wavelength of the incident wave.
        exp_name (str): The name of the experiment (default is 'optics').
        nruns (int): The number of runs of the epde_discovery (default is 1).
        solve_equations (bool): Flag to solve equations (default is True).
        pop_size: The population size for EPDE.
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

    m_grid_training, m_grid_test, poynting_vec_training, poynting_vec_test = get_split_data(r0)
    results_dir = get_results_dir(exp_name)
    save_split_exp_data(r0, m_grid_training, m_grid_test, poynting_vec_training, poynting_vec_test, results_dir)

    for run in range(nruns):
        start_run(r0, wave_length, run, m_grid_training / wave_length,
                  m_grid_test / wave_length, poynting_vec_training,
                  poynting_vec_test, pop_size, factors_max_number,
                  poly_order, training_epde_epochs, variable_names,
                  max_deriv_order, equation_terms_max_number, data_fun_pow,
                  solve_equations, training_tedeous_epochs, results_dir)


def save_solutions_visualization(r0_list: list, exp_name: str, wave_length: float, pop_size: int,
                                 nruns: int, add_legend: bool, add_training_data: bool) -> None:
    """
    Save visualizations of solutions for a given experiment and parameters.

    Args:
        r0_list (list): List containing radius values.
        exp_name (str): Name of the experiment.
        wave_length (float): Wavelength value.
        pop_size (int): The population size for EPDE.
        nruns (int): The number of runs of the epde_discovery.
        add_legend (bool): Flag to add a legend to the visualization.
        add_training_data (bool): Flag to include training data in the visualization.

    Returns:
        None
    """

    results_dir = get_results_dir(exp_name)

    for r0, i, run in itertools.product(r0_list, range(pop_size), range(nruns)):
        grid_training = np.genfromtxt(results_dir / 'split exp data' / f'grid_training_{r0}.txt') / wave_length
        grid_test = np.genfromtxt(results_dir / 'split exp data' / f'grid_test_{r0}.txt') / wave_length

        poynting_vec_training = np.genfromtxt(results_dir / 'split exp data' / f'poynting_vec_training_{r0}.txt')
        poynting_vec_test = np.genfromtxt(results_dir / 'split exp data' / f'poynting_vec_test_{r0}.txt')

        if not Path(
            results_dir
            / 'solutions data'
            / f'sln_data_training_{r0}_{i}_{run}.txt'
        ).exists():
            continue

        pred_solution_training = torch.from_numpy(np.genfromtxt(results_dir / 'solutions data' /
                                                                f'sln_data_training_{r0}_{i}_{run}.txt'))
        pred_solution_test = torch.from_numpy(np.genfromtxt(results_dir / 'solutions data' /
                                                            f'sln_data_test_{r0}_{i}_{run}.txt'))
        draw_solution(r0, wave_length, i, run, grid_training, grid_test, poynting_vec_training, poynting_vec_test,
                      pred_solution_training, pred_solution_test, results_dir, save_solutions=True,
                      add_legend=add_legend, add_training_data=add_training_data)
