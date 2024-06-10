import numpy as np
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
