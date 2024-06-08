import epde.interface.interface as epde_alg
import numpy as np
from epde.interface.prepared_tokens import GridTokens, CacheStoredTokens
from epde.interface.interface import EpdeSearch


def set_de_params(epde_search_obj: epde_alg.EpdeSearch, pop_size: int, training_epochs: int) -> None:
    """
    Set the MOEADD parameters for the EPDE search object.

    Args:
        epde_search_obj: The EPDE search object to set parameters for.
        pop_size: The population size.
        training_epochs: The number of training epochs.

    Returns:
        None
    """

    epde_search_obj.set_moeadd_params(population_size=pop_size, training_epochs=training_epochs)


def get_polynomial_family(tensor: np.ndarray, order: int, token_type='polynomials') -> CacheStoredTokens:
    """
    Get family of tokens for polynomials of orders from second up to the order argument.

    Args:
        tensor: The input tensor for generating polynomial tokens.
        order: The maximum order of polynomials to generate tokens for.
        token_type: The type of tokens to be generated (default is 'polynomials').

    Returns:
        CacheStoredTokens: An object containing the generated polynomial tokens.
    """

    assert order > 1
    labels = [f'I^{idx + 1}' for idx in range(1, order)]
    tensors = {label: tensor ** (idx + 2) for idx, label in enumerate(labels)}
    return CacheStoredTokens(token_type=token_type,
                             token_labels=labels,
                             token_tensors=tensors,
                             params_ranges={'power': (1, 1)},
                             params_equality_ranges=None, meaningful=True)


def set_epde_preprocessor(epde_search_obj: EpdeSearch, use_ann: bool = False) -> None:
    """
    Set the preprocessor for an EpdeSearch object based on the specified configuration.

    Args:
        epde_search_obj: The EpdeSearch object to set the preprocessor for.
        use_ann: A boolean indicating whether to use the ANN preprocessor (default is False).

    Returns:
        None
    """

    if use_ann:
        epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
                                         preprocessor_kwargs={'epochs_max': 50000})
    else:
        epde_search_obj.set_preprocessor(default_preprocessor_type='poly',
                                         preprocessor_kwargs={'use_smoothing': False, 'sigma': 1,
                                                              'polynomial_window': 3,
                                                              'poly_order': 3})


def epde_discovery(grid: np.ndarray, poynting_vec: np.ndarray, pop_size: int = 5,
                   factors_max_number: int = 1, poly_order: int = 4, training_epochs: int = 100,
                   variable_names: [str] = None, max_deriv_order: (int, ) = (2,),
                   equation_terms_max_number: int = 5, data_fun_pow: int = 1,
                   use_ann: bool = False, derivs: [[np.ndarray]] = None) -> EpdeSearch:
    """
    Perform EPDE discovery to find equations describing the relationship between grid and Poynting vector.

    Args:
        grid: The grid data.
        poynting_vec: The Poynting vector data.
        pop_size: The population size (default is 5).
        factors_max_number: The maximum number of factors in the equations (default is 1).
        poly_order: The order of polynomials to generate tokens for (default is 4).
        training_epochs: The number of training epochs (default is 100).
        variable_names: The names of variables (default is ['I']).
        max_deriv_order: The maximum derivative order (default is (2,)).
        equation_terms_max_number: The maximum number of terms in the equations (default is 5).
        data_fun_pow: The highest power of derivative-like token in the equation (default is 1).
        use_ann: Flag to indicate whether to use ANN preprocessor (default is False).
        derivs: The derivatives data (default is None).

    Returns:
        EpdeSearch: The object containing the discovered equations.
    """

    if variable_names is None:
        variable_names = ['I', ]
    dimensionality = poynting_vec.ndim - 1
    epde_search_obj = epde_alg.EpdeSearch(use_solver=False, dimensionality=dimensionality,
                                          coordinate_tensors=[grid, ])

    set_epde_preprocessor(epde_search_obj, use_ann=use_ann)

    epde_search_obj.set_moeadd_params(population_size=pop_size, training_epochs=training_epochs)

    custom_grid_tokens = GridTokens(dimensionality=dimensionality)
    polynomial_tokens = get_polynomial_family(poynting_vec, poly_order)

    kwargs = {'data': [poynting_vec, ], 'variable_names': variable_names,
              'max_deriv_order': max_deriv_order, 'derivs': [derivs, ],
              'equation_terms_max_number': equation_terms_max_number, 'data_fun_pow': data_fun_pow,
              'additional_tokens': [polynomial_tokens, custom_grid_tokens if derivs is None else polynomial_tokens],
              'equation_factors_max_number': factors_max_number, 'eq_sparsity_interval': (1e-12, 1e-4)}
    epde_search_obj.fit(**kwargs)

    epde_search_obj.equations(only_print=True, num=1)
    return epde_search_obj
