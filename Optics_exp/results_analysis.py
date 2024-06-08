import numpy as np
import itertools
import pandas as pd
import re
from pathlib import Path


def get_results_dir(exp_name: str) -> Path:
    results_dir = Path.cwd() / 'results' / f'results_{exp_name}'
    if not (Path(results_dir).exists()):
        Path(results_dir).mkdir()
    return results_dir


def get_eq_terms_from_string(line: str) -> [str]:
    """
    Extracts and splits equation terms from a given EPDE-line of an equation.

    Args:
        line (str): The line containing the equation terms in EPDE-form.

    Returns:
        list: A list of extracted equation terms.

    Raises:
        None
    """
    eq_terms = (line.replace('\n', '').replace(' ', '').
                replace('{power:1.0}', '').replace('=', '=1.0*').
                replace('x0', 'H').replace('t{power:1.0,dim:0.0}', 'H'))
    return re.split("\+|\=", eq_terms)


def get_coefs_from_terms(eq_terms: [str]) -> dict:
    """
    Extracts coefficients from the provided equation terms and returns them as a dictionary.

    Args:
        eq_terms: List of equation terms.

    Returns:
        Dictionary containing the extracted coefficients and the corresponding terms.
    """

    coefs = {}
    for term in eq_terms:
        factors = term.split('*')
        if float(factors[0]) != 0:
            if len(factors) == 1:
                coefs["C"] = float(factors[0])
            elif len(factors) == 2:
                coefs[str(factors[1])] = float(factors[0])
            else:
                coefs["*".join(factors[1:])] = float(factors[0])
    return coefs


def get_rmse(exp_name: str, r0: int | float, i: int, run: int) -> float:
    """
    Calculates the Root Mean Square Error (RMSE) between the data from the fitted model and test data for
    a specific experiment.

    Args:
        exp_name: Name of the experiment.
        r0: The radius value.
        i: The index of the equation in the population value.
        run: The run number for this value of r0.

    Returns:
        The calculated RMSE as a float.
    """

    results_dir_name = get_results_dir(exp_name)

    sln_data_file_path = results_dir_name / 'solutions data' / f'sln_data_test_{r0}_{i}_{run}.txt'
    sln = np.genfromtxt(sln_data_file_path, delimiter=',')

    test_data_file_path = results_dir_name / 'split exp data' / f'poynting_vec_test_{r0}.txt'
    test_data = np.genfromtxt(test_data_file_path, delimiter=',')

    return np.sqrt(np.mean((sln - test_data) ** 2))


def read_eqn(exp_name: str, eqn_id: str) -> dict | None:
    """
    Reads and parses an equation from a file.

    Args:
        exp_name (str): The experiment name.
        eqn_id (str): The ID of the equation file to read.

    Returns:
        dict | None: A dictionary representing the parsed equation terms and RMSE value,
        or None if there is an issue opening the file.

    Raises:
        None
    """
    results_dir_name = get_results_dir(exp_name)
    eqn_file_path = results_dir_name / 'text equations' / eqn_id
    if not eqn_file_path.exists():
        return None

    with eqn_file_path.open() as file:
        lines = file.readlines()
        eq_terms = get_eq_terms_from_string(lines[0])
    coefs = get_coefs_from_terms(eq_terms)

    r0, i, run = eqn_id.split('_')[1:]
    r0 = float(r0) if '.' in r0 else int(r0)
    i = int(i)
    run = int(run[:len(run) - 4])

    coefs['rmse'] = get_rmse(exp_name, r0, i, run)
    return coefs


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
    terms[0] = "$$" + terms[0]
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


def save_total_results_latex_form(r0_list: [int | float], exp_name: str,  pop_size: int,
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
