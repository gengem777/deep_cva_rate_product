import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xvaEquation as eqn
from FiccSolver import BSDESolver
from FiccXvaSolver import FiccXvaSolver
import RecursiveEquation as receqn
import munch
import math
import pandas as pd


if __name__ == "__main__":
    dim = 1 # dimension of brownian motion
    P = 2048  # number of outer Monte Carlo Loops
    batch_size = 128
    total_time = 1.0
    swap_time = 2.0
    num_time_interval = 100
    sigma = 0.002
    kappa = 0.04
    theta = 1.0
    r_init = 0.03
    exact = 0.0  # fair value of IRS is 0
    strike = 0.001
    config = {

        "eqn_config": {
            "_comment": "a basket call option",
            "eqn_name": "InterestRateSwap",
            "total_time": total_time,
            "swap_time": swap_time,
            "dim": dim,
            "num_time_interval": num_time_interval,
            "strike": strike,
            "r_init": r_init,
            "sigma": sigma,
            "kappa": kappa,
            "theta": theta,
        },

        "net_config": {
            "y_init_range": [-2.0, 2.0],  # [154.37,165.41], #set to None when not sure
            "num_hiddens": [dim + 10, dim + 10],
            "lr_values": [5e-2, 5e-3],  # [5e-1,5e-2, 5e-3],
            "lr_boundaries": [2000],  # [1000,2000],
            "num_iterations": 6000,
            "batch_size": batch_size,
            "valid_size": 128,
            "logging_frequency": 100,
            "dtype": "float64",
            "verbose": True
        }
    }
    config = munch.munchify(config)
    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    tf.keras.backend.set_floatx(config.net_config.dtype)

    # apply algorithm 1
    bsde_solver = BSDESolver(config, bsde)
    training_history = bsde_solver.train()

    # Simulate the BSDE after training - MtM scenarios
    simulations = bsde_solver.model.simulate_path(bsde.sample(P))
    def zcb(t, T):
        r = r_init
        B = (1 - math.exp(-kappa * (T - t))) / kappa
        A = math.exp((B - T + t)*(kappa**2 * theta - sigma**2/2)/kappa**2 +\
                     (sigma*B)**2/(4*kappa))
        result = A * math.exp(-B * r)
        return result
    exact = zcb(0, total_time) - zcb(0, swap_time) - (swap_time - total_time)*strike*zcb(0, swap_time)
    print(simulations[0, 0, 0], exact, np.abs((simulations[0, 0, 0] - exact)))
    if np.abs((simulations[0, 0, 0] - exact)) < 0.0005:
        print('clean value test passed')
    else:
        print('clean value test failed')
        print('Test Failed')
        raise ValueError("clean value error")

    configIRSCVA = {
        "eqn_config": {
            "_comment": "BCVA on a basket call",
            "eqn_name": "InterestRateSwapCVA",
            "dim": dim,
            "total_time": total_time,
            "swap_time": swap_time,
            "num_time_interval": num_time_interval,
            "strike": strike,
            "r_init": r_init,
            "sigma": sigma,
            "kappa": kappa,
            "theta": theta,
            "recoveryC": 0.0,
            "lambdaC": 0.0,
            "recoveryB": 0.0,
            "lambdaB": 0.1,
            "clean_value": bsde,
            "clean_value_model": bsde_solver.model
        },
        "net_config": {
            "y_init_range": [0, 10],
            "num_hiddens": [dim + 10, dim + 10],
            "lr_values": [5e-2, 5e-3],
            "lr_boundaries": [2000],
            "num_iterations": 4000,
            "batch_size": batch_size,
            "valid_size": 128,
            "logging_frequency": 100,
            "dtype": "float64",
            "verbose": True
        }
    }
    configIRSCVA = munch.munchify(configIRSCVA)
    cvabsde = getattr(receqn, configIRSCVA.eqn_config.eqn_name)(configIRSCVA.eqn_config) #BCVA
    tf.keras.backend.set_floatx(configIRSCVA.net_config.dtype)

    # apply algorithm 3
    cva_solver = FiccXvaSolver(configIRSCVA, cvabsde)
    # loss: 1.7611e-01, Y0: 6.9664e-01,
    irscva_training_history = cva_solver.train()

    cva_simulations = cva_solver.model.simulate_path(cvabsde.sample(P))
    # (0.699395244753698, [0.6903630282972714, 0.7084274612101246])
    mc_result = cvabsde.monte_carlo(100000)
    print(mc_result[0])
    print(cva_simulations[0,0,0])
    ucb = mc_result[1][1]
    lcb = mc_result[1][0]
    # time_stamp = np.linspace(0, 1, num_time_interval + 1)
    # fig = plt.figure()
    # plt.plot(time_stamp, np.transpose(np.mean(cva_simulations[:,0,:], axis=0)), 'b', label='CVA')
    # plt.xlabel('t')
    # plt.legend()
    # plt.show()
    if cva_simulations[0,0,0] >= lcb and cva_simulations[0,0,0] <= ucb:
        print('cva test passed')
        print('Test Passed')
    else:
        print('cva test failed')
        print('Test failed')
        raise ValueError("cva value error")