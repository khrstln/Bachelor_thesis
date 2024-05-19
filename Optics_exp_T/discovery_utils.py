import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import epde
from data_utils import read_data
import epde.interface.interface as epde_alg
from epde.interface.prepared_tokens import TrigonometricTokens, \
    CustomEvaluator, CustomTokens, GridTokens, CacheStoredTokens


def get_polynomial_family(tensor, order, token_type='polynomials'):
    """
    Get family of tokens for polynomials of orders from second up to order argument.
    """
    assert order > 1
    labels = [f'I^{idx + 1}' for idx in range(1, order)]
    tensors = {label : tensor ** (idx + 2) for idx, label in enumerate(labels)}
    return CacheStoredTokens(token_type=token_type,
                                token_labels=labels,
                                token_tensors=tensors,
                                params_ranges={'power': (1, 1)},
                                params_equality_ranges=None, meaningful=True)


def epde_discovery(t, x, boundary=0, use_ann=False, derivs=None):
    dimensionality = x.ndim - 1
    epde_search_obj = epde_alg.EpdeSearch(use_solver=False, dimensionality=dimensionality, boundary=boundary,
                                          coordinate_tensors=[t, ])
    if use_ann:
        epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',  # use_smoothing = True poly
                                         preprocessor_kwargs={'epochs_max': 50000})  #
    else:
        epde_search_obj.set_preprocessor(default_preprocessor_type='poly',  # use_smoothing = True poly
                                         preprocessor_kwargs={'use_smoothing': False, 'sigma': 1,
                                                              'polynomial_window': 3,
                                                              'poly_order': 3})  # 'epochs_max' : 10000})#
    pop_size = 4
    factors_max_number = 1
    epde_search_obj.set_moeadd_params(population_size=pop_size, training_epochs=100)
    # factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}  # 1 factor with P = 0.65, 2 with P = 0.35
    custom_grid_tokens = GridTokens(dimensionality=dimensionality)
    polynomial_tokens = get_polynomial_family(x, 4, token_type='polynomials')
    if derivs is None:
        epde_search_obj.fit(data=[x,], variable_names=['I',], max_deriv_order=(2,),
                            equation_terms_max_number=5, data_fun_pow=1,
                            additional_tokens=[polynomial_tokens],
                            equation_factors_max_number=factors_max_number,
                            eq_sparsity_interval=(1e-12, 1e-4))
    else:
        epde_search_obj.fit(data=[x, ], variable_names=['I',], max_deriv_order=(2,),
                            derivs=[derivs, ],
                            equation_terms_max_number=5, data_fun_pow=1,
                            additional_tokens=[polynomial_tokens],
                            equation_factors_max_number=factors_max_number,
                            eq_sparsity_interval=(1e-12, 1e-4))
    epde_search_obj.equations(only_print=True, num=1)
    """
    Having insight about the initial ODE structure, we are extracting the equation with complexity of 5
    In other cases, you should call sys.equation_search_results(only_print = True),
    where the algorithm presents Pareto frontier of optimal equations.
    """
    return epde_search_obj

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device, check_device
from tedeous.models import mat_model
import torch

def solver_solution(eq, I_lambda, m_grid):
    solver_device('cpu')
    mode = 'autograd'
    coord_list = [m_grid]
    coord_list = torch.tensor(coord_list)
    grid = coord_list.reshape(-1,1).float()

    ##Domain class for doamin initialization
    domain = Domain()
    domain.variable('y', grid, None)

    boundaries = Conditions()
    ##initial cond
    x = domain.variable_dict['y']
    boundaries.dirichlet({'y': 0}, value=-1.0)
    boundaries.dirichlet({'y': m_grid[-1]}, value=I_lambda[-1])

    equation = Equation()
    equation.add(eq)

    img_dir = os.path.join(os.path.dirname(__file__), 'optics_intermediate')

    if mode in ('NN', 'autograd'):
        net = torch.nn.Sequential(
            torch.nn.Linear(1, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 1)
        )
    else:
        net = mat_model(domain, equation)

    model = Model(net, domain, equation, boundaries)
    model.compile(mode, lambda_operator=1, lambda_bound=40)
    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=1000,
                                         patience=3,
                                         randomize_parameter=1e-5,
                                         info_string_every=1000)
    cb_plots = plot.Plots(save_every=1000, print_every=None, img_dir=img_dir)
    optimizer = Optimizer('Adam', {'lr': 1e-3})
    model.train(optimizer, 10000, save_model=False, callbacks=[cb_es, cb_plots])
    pred_solution = check_device(net(grid)).reshape(-1)
    return pred_solution

