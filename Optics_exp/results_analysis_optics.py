import numpy as np
import itertools
import os
import sys
import pandas as pd
import re


def get_eq_terms(line: str):
    """
    Extracts and splits equation terms from a given EPDE-line of an equation.

    Args:
        line (str): The line containing the equation terms in EPDE-form.

    Returns:
        list: A list of extracted equation terms.

    Raises:
        None
    """
    eq_terms = line.replace("\n", "").replace(" ", "").replace("{power:1.0}", "").replace("=", "=1.0*").replace(
        'x0''y')
    return re.split("\+|\=", eq_terms)


def read_eqn(dir_name: str, eqn_id: str) -> dict | None:
    """
    Reads and parses an equation from a file.

    Args:
        dir_name (str): The directory name where the equation file is located.
        eqn_id (str): The ID of the equation file to read.

    Returns:
        dict | None: A dictionary representing the parsed equation terms, or None if there is an issue opening the file.

    Raises:
        None
    """

    file_path = os.path.join(dir_name, eqn_id)
    try:
        file = open(file_path)
    except Exception:
        return
    lines = file.readlines()
    eq_terms = get_eq_terms(lines[0])

    eqn = {}
    for term in eq_terms:
        factors = term.split('*')
        if float(factors[0]) != 0:
            if len(factors) == 1:
                eqn["C"] = float(factors[0])
            elif len(factors) == 2:
                eqn[str(factors[1])] = float(factors[0])
            else:
                eqn["*".join(factors[1:])] = float(factors[0])
    return eqn


sys.path.append('../')
sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__))))
results_dir = os.path.join(os.path.dirname(__file__), r'results_optics\text equations')
eqn_list = []

for r0_fix in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    eqn_list.extend(
        f'eqn_{r0_fix}_{i}_{j}.txt'
        for i, j in itertools.product(range(4), range(1))
    )
read_eq_dict = {
    eq: read_eqn(results_dir, eq)
    for eq in eqn_list
    if read_eqn(results_dir, eq) is not None
}
df = pd.DataFrame(read_eq_dict).transpose().fillna(0)

print(df)
df.to_csv(r'results_optics\total_results_optics.csv')

with open(os.path.join(os.path.dirname(__file__), r'results_optics\Equations.md'), 'w') as eq_file:
    for eq_name in df.index:
        coefs = np.array(df.loc[eq_name])
        terms = [fr'{coefs[i]:.3f} \cdot ' + df.columns[i].replace('C', '') for i in range(len(coefs)) if
                 coefs[i] not in [0.0] \
                 and df.columns[i] != 'dI/dy']
        terms[0] = "$$" + terms[0]
        eq_name = eq_name.replace('.txt', '')
        params = eq_name.split('_')
        params = {"$r_0$ = ": params[1], "index = ": params[2], "nrun = ": params[3]}
        res = "$r_0$ = " + params["$r_0$ = "] + ", index = " + params["index = "] + ", nrun = " + params[
            "nrun = "] + ": " + " + ".join(terms) + " = dI/dy$$" + "\n\n"
        res = res.replace('+ -', '- ')
        res = res.replace('dI/dy', r'\frac{dI}{dy}')
        res = res.replace(r'\cdot  ', '')
        eq_file.write(res)
