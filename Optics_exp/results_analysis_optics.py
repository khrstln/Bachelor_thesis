import numpy as np

import os
import sys
import pandas as pd
import re

def read_eqn(results_dir,eqn_id):
    file_path = os.path.join(results_dir,eqn_id)
    try:
        file = open(file_path)
    except:
        return
    lines = file.readlines()
    terms = lines[0].replace("\n","").replace(" ","").replace("{power:1.0}","").replace("=","=1.0*").replace('x0', 'y')
    terms = re.split("\+|\=",terms)
    eqn = {}
    for term in terms:
        term1=term.split('*')
        if float(term1[0]) != 0:
            if len(term1) == 1:
                eqn["C"] = float(term1[0])
            elif len(term1) == 2:
                eqn[str(term1[1])] = float(term1[0])
            else:
                s1 = "*"
                eqn[s1.join(term1[1:])] = float(term1[0])
    return eqn

sys.path.append('../')
sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname( __file__ ))))
results_dir = os.path.join(os.path.dirname( __file__ ), r'results_optics\text equations')
eqn_list = []
for r0_fix in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
# for r0_fix in [0.5]:
    for i in range(4):
        # if r0_fix == 0.2 and i == 3:
        #     continue
        for j in range(1):
            eqn_list.append(f'eqn_{r0_fix}_{i}_{j}.txt')
read_eq_dict={}
for eq in eqn_list:
    if read_eqn(results_dir,eq) is None:
        continue
    read_eq_dict[eq]=read_eqn(results_dir,eq)
df = pd.DataFrame(read_eq_dict).transpose().fillna(0)

print(df)
# if 'd^2I/dy^2' in df.columns:
#     divisor=df['d^2I/dy^2'].values
#     for col in df.columns:
#         df[col]=df[col].values/divisor
df.to_csv(r'results_optics\total_results_optics.csv')

with open(os.path.join(os.path.dirname( __file__ ), r'results_optics\Equations.md'), 'w') as eq_file:
    for eq_name in df.index:
        coefs = np.array(df.loc[eq_name])
        terms = [fr'{coefs[i]:.3f} \cdot ' + df.columns[i].replace('C', '') for i in range(len(coefs)) if coefs[i] not in [0.0] \
                 and df.columns[i] != 'dI/dy']
        terms[0] = "$$" + terms[0]
        eq_name = eq_name.replace('.txt', '')
        params = eq_name.split('_')
        params = {"$r_0$ = " : params[1], "index = " : params[2], "nrun = ": params[3]}
        res = "$r_0$ = " + params["$r_0$ = "] + ", index = " + params["index = "] + ", nrun = " + params["nrun = "] + ": " + " + ".join(terms) + " = dI/dy$$" + "\n\n"
        res = res.replace('+ -', '- ')
        res = res.replace('dI/dy', r'\frac{dI}{dy}')
        res = res.replace(r'\cdot  ', '')
        eq_file.write(res)

