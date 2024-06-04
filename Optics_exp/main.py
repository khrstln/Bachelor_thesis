from optics_exp import optics_exp
from tedeous.device import solver_device

if __name__ == '__main__':
    solver_device('cpu')

    # for r0_fix in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    #     optics_exp(r0_fix, exp_name='optics')

    # for r0_fix in [0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    #     optics_exp(r0_fix, exp_name='optics')

    for r0_fix in [0.1]:
        optics_exp(r0_fix, exp_name='optics')
