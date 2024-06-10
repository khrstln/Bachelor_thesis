from interface import *
from tedeous.device import solver_device

if __name__ == '__main__':
    solver_device('gpu')
    wave_length = 0.5  # The wavelength value in micrometers
    r0_list = [r / 10 for r in range(1, 10)]  # Radius values of the dielectric inclusions in 2D supercell
    # model of the inhomogeneous layer value in micrometers
    exp_name = 'test_10.06.24_1'  # The name of the experiment
    pop_size = 6
    nruns = 1

    # for r0_fix in r0_list:
    #     start_exp(r0_fix, wave_length, exp_name=exp_name, nruns=nruns, solve_equations=True,
    #               pop_size=pop_size, factors_max_number=1, poly_order=4, variable_names=['I'],
    #               max_deriv_order=(2,), equation_terms_max_number=5,
    #               data_fun_pow=1, training_epde_epochs=5, training_tedeous_epochs=100)

    save_total_results_csv(r0_list, exp_name, pop_size, nruns)
    save_total_results_latex_form(r0_list, exp_name, pop_size, nruns)
    save_solutions_visualization(r0_list, exp_name, wave_length, pop_size, nruns, True, False)
