import epde.interface.interface as epde_alg
import numpy as np
from epde.interface.prepared_tokens import GridTokens, CacheStoredTokens


def set_de_params(epde_search_obj: epde_alg.EpdeSearch, pop_size: int, training_epochs: int):
    epde_search_obj.set_moeadd_params(population_size=pop_size, training_epochs=training_epochs)


def get_polynomial_family(tensor: np.ndarray, order: int, token_type='polynomials'):
    """
    Get family of tokens for polynomials of orders from second up to order argument.
    """
    assert order > 1
    labels = [f'I^{idx + 1}' for idx in range(1, order)]
    tensors = {label: tensor ** (idx + 2) for idx, label in enumerate(labels)}
    return CacheStoredTokens(token_type=token_type,
                             token_labels=labels,
                             token_tensors=tensors,
                             params_ranges={'power': (1, 1)},
                             params_equality_ranges=None, meaningful=True)


def epde_discovery(grid: np.ndarray, poynting_vec: np.ndarray, pop_size: int = 5,
                   factors_max_number: int = 1, poly_order: int = 4, training_epochs: int = 100,
                   variable_names: [str] = None, max_deriv_order: tuple = (2,),
                   equation_terms_max_number: int = 5, data_fun_pow: int = 1,
                   use_ann: bool = False, derivs: [[np.ndarray]] = None):
    if variable_names is None:
        variable_names = ['I', ]
    dimensionality = poynting_vec.ndim - 1
    epde_search_obj = epde_alg.EpdeSearch(use_solver=False, dimensionality=dimensionality,
                                          coordinate_tensors=[grid, ])
    if use_ann:
        epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
                                         preprocessor_kwargs={'epochs_max': 50000})
    else:
        epde_search_obj.set_preprocessor(default_preprocessor_type='poly',
                                         preprocessor_kwargs={'use_smoothing': False, 'sigma': 1,
                                                              'polynomial_window': 3,
                                                              'poly_order': 3})
    epde_search_obj.set_moeadd_params(population_size=pop_size, training_epochs=training_epochs)
    custom_grid_tokens = GridTokens(dimensionality=dimensionality)
    polynomial_tokens = get_polynomial_family(poynting_vec, poly_order)
    if derivs is None:
        epde_search_obj.fit(data=[poynting_vec, ], variable_names=variable_names, max_deriv_order=max_deriv_order,
                            equation_terms_max_number=equation_terms_max_number, data_fun_pow=data_fun_pow,
                            additional_tokens=[polynomial_tokens, custom_grid_tokens],
                            equation_factors_max_number=factors_max_number,
                            eq_sparsity_interval=(1e-12, 1e-4))
    else:
        epde_search_obj.fit(data=[poynting_vec, ], variable_names=variable_names, max_deriv_order=max_deriv_order,
                            derivs=[derivs, ],
                            equation_terms_max_number=equation_terms_max_number, data_fun_pow=data_fun_pow,
                            additional_tokens=[polynomial_tokens],
                            equation_factors_max_number=factors_max_number,
                            eq_sparsity_interval=(1e-12, 1e-4))
    epde_search_obj.equations(only_print=True, num=1)
    return epde_search_obj
