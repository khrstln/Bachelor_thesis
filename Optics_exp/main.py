from optics_exp import optics_exp
from tedeous.device import solver_device

if __name__ == '__main__':
    solver_device('cpu')

    for r0_fix in [r / 10 for r in range(1, 11)]:
        optics_exp(r0_fix, exp_name='optics', nruns=1, solve_equations=True,
                   factors_max_number=1, poly_order=4, variable_names=['I'],
                   max_deriv_order=(2,), equation_terms_max_number=5,
                   data_fun_pow=1, training_epde_epochs=100, training_tedeous_epochs=10000)
