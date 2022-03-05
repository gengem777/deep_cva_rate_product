import pytest
import os
from FiccSolver import BSDESolver
from xvaEquation import InterestRateSwap
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xvaEquation as eqn
from FiccSolver import BSDESolver, FeedForwardSubNet
from FiccXvaSolver import FiccXvaSolver
import RecursiveEquation as receqn
import munch
import math


class TestSOLVER:
    def test_NonSharedModel_V1(self):
        """
        input zeros and check whether the outputs are also zeros thoughout the whole model
        """
        # set params
        config_IRS = {

            "eqn_config": {
                "_comment": "a basket call option",
                "eqn_name": "InterestRateSwap",
                "total_time": 1.0,
                "swap_time": 2.0,
                "dim": 1,
                "num_time_interval": 100,
                "strike": 0.00,
                "r_init": 0.03,
                "sigma": 0.002,
                "kappa": 0.04,
                "theta": 1.0,
            },

            "net_config": {
                "y_init_range": [-2.0, 2.0],  # [154.37,165.41], #set to None when not sure
                "num_hiddens": [5, 5],
                "lr_values": [5e-2, 5e-3],  # [5e-1,5e-2, 5e-3],
                "lr_boundaries": [2000],  # [1000,2000],
                "num_iterations": 1000,
                "batch_size": 128,
                "valid_size": 128,
                "logging_frequency": 100,
                "dtype": "float64",
                "verbose": False
            }
        }

        config = munch.munchify(config_IRS)
        bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
        tf.keras.backend.set_floatx(config.net_config.dtype)
        bsde_solver = BSDESolver(config, bsde)
        model = bsde_solver.model
        model.y_init = tf.Variable(np.random.uniform(low=0.0,
                                                    high=0.0,
                                                    size=[1]), dtype=config.net_config.dtype
                                  )
        model.z_init = tf.Variable(np.random.uniform(low=0.0, high=0.0,
                                                    size=[1, config.eqn_config.dim]), dtype=config.net_config.dtype
                                  )
        dw, r, B, P = bsde_solver.bsde.sample(128)
        dw_0 = tf.zeros_like(dw)
        inputs = dw_0, r, B, P
        y_terminal = model(inputs, training=False)
        y_0 = tf.zeros_like(y_terminal)
        diff = y_terminal.numpy() - y_0.numpy()
        assert np.sum(diff**2) == 0

    def test_NonSharedModel_V2(self):
        '''
        reduce the network to a linear function and time_step to 2 and check whether the output of the model is equal to
        the one calculated by hand (analytical one)
        '''
        # set params
        config_IRS = {

            "eqn_config": {
                "_comment": "a FRA",
                "eqn_name": "InterestRateSwap",
                "eqn_type": "ficc", #equity or ficc
                "total_time": 1.0,
                "swap_time": 1.0,
                "dim": 1,
                "num_time_interval": 2,
                "strike": 0.001,
                "r_init": 0.03,
                "sigma": 0.002,
                "kappa": 0.04,
                "theta": 0.1,
            },

            "net_config": {
                "y_init_range": [1.0, 1.0],  # [154.37,165.41], #set to None when not sure
                "num_hiddens": [],
                "lr_values": [5e-2, 5e-3],  # [5e-1,5e-2, 5e-3],
                "lr_boundaries": [2000],  # [1000,2000],
                "num_iterations": 6000,
                "batch_size": 128,
                "valid_size": 128,
                "logging_frequency": 100,
                "dtype": "float64",
                "verbose": True
            }
        }
        config = munch.munchify(config_IRS)
        bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
        tf.keras.backend.set_floatx(config.net_config.dtype)
        bsde_solver = BSDESolver(config, bsde)
        model = bsde_solver.model
        y_0 = model.y_init
        z_0 = model.z_init
        subnet = FeedForwardSubNet(config, 1)
        subnet.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                center=False,
                scale=False,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.0),
                gamma_initializer=tf.random_uniform_initializer(1.0, 1.0)
            )
            for _ in range(len(config.net_config.num_hiddens) + 2)]
        subnet.dense_layers[-1] = tf.keras.layers.Dense(1, kernel_initializer='ones',use_bias=True,activation=None, )
        model.subnet = [subnet]
        dw = np.array([[[0.1, 0.2]]])
        r = np.array([[[0.01, 0.02, 0.01]]])
        B = np.array([[[1.0, 1.0, 1.0]]])
        P = np.array([[[0.8, 0.9, 1.0]]])
        input = dw, r, B, P
        y_terminal = model(input, training=False)

        #analytical form
        print('y_0', y_0, z_0)
        y = y_0 + r[:,:,0]*y_0*0.5 + z_0*dw[:,:,0]
        y = y + r[:,:,1]*y*0.5 + (1.0*P[:,:,1]+0.0)*dw[:,:,1]
        y_terminal = y_terminal.numpy()
        y = y.numpy()
        assert np.abs(y - y_terminal) < 0.0001



    def test_FeedForwardSubNet(self):
        '''
        reduce the single block to an affine function and check whether the output of the single network is as same as
        what is calculated by hand (analytical one)
        '''
        # set params
        config_IRS = {

            "eqn_config": {
                "_comment": "a basket call option",
                "eqn_name": "InterestRateSwap",
                "eqn_type": "ficc",  # equity or ficc
                "total_time": 1.0,
                "swap_time": 1.0,
                "dim": 1,
                "num_time_interval": 1,
                "strike": 0.001,
                "r_init": 0.03,
                "sigma": 0.002,
                "kappa": 0.04,
                "theta": 0.1,
            },

            "net_config": {
                "y_init_range": [1.0, 1.0],  # [154.37,165.41], #set to None when not sure
                "num_hiddens": [],
                "lr_values": [5e-2, 5e-3],  # [5e-1,5e-2, 5e-3],
                "lr_boundaries": [2000],  # [1000,2000],
                "num_iterations": 6000,
                "batch_size": 128,
                "valid_size": 128,
                "logging_frequency": 100,
                "dtype": "float64",
                "verbose": True
            }
        }
        config = munch.munchify(config_IRS)
        model = FeedForwardSubNet(config, 1)
        model.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                center=False,
                scale=False,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.0),
                gamma_initializer=tf.random_uniform_initializer(1.0, 1.0)
            )
            for _ in range(len(config.net_config.num_hiddens) + 2)]

        for _ in range(100):
            x = tf.ones((1, 1))
            y = model(x).numpy()
            weight = model.dense_layers[-1].get_weights()
            a = weight[0]
            b = weight[1]
            y_true = (a*x+b).numpy()
            assert np.abs(y - y_true) < 0.0001

    def test_grad(self):
        '''
        take a simple function as an example and check whether the gradient calculated via the tested function is as same
        as the derivative calculated by math.
        '''
        x = tf.Variable(np.array([1.0]))
        with tf.GradientTape(persistent=True) as tape:
            y = x **2 + 2*x +1
        grad = tape.gradient(y, x)
        del tape
        assert grad == np.array([4.0])

        with tf.GradientTape(persistent=True) as tape:
            y = tf.exp(x)
        grad = tape.gradient(y, x)
        del tape
        assert grad == np.array(np.e)




    def test_train(self):
        '''
        train the model and check whether the training and validating loss tend to zero on both training and validating set
        '''
        #set params
        config_IRS = {

            "eqn_config": {
                "_comment": "FRA",
                "eqn_name": "InterestRateSwap",
                "eqn_type": "ficc",  # equity or ficc
                "total_time": 1.0,
                "swap_time": 2.0,
                "dim": 1,
                "num_time_interval": 100,
                "strike": 0.0,
                "r_init": 0.03,
                "sigma": 0.002,
                "kappa": 0.04,
                "theta": 1.0,
            },

            "net_config": {
                "y_init_range": [-1.0, 1.0],  # [154.37,165.41], #set to None when not sure
                "num_hiddens": [5, 5], #set a tiny network
                "lr_values": [5e-2, 5e-3],  # [5e-1,5e-2, 5e-3],
                "lr_boundaries": [2000],  # [1000,2000],
                "num_iterations": 3000,
                "batch_size": 128,
                "valid_size": 128,
                "logging_frequency": 200,
                "dtype": "float64",
                "verbose": True
            }
        }
        config = munch.munchify(config_IRS)
        bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
        tf.keras.backend.set_floatx(config.net_config.dtype)
        bsde_solver = BSDESolver(config, bsde)
        training_history = bsde_solver.train()
        #generate validate data
        LOSS_ERROR = 1e-6
        GRAD_ERROR = 1e-5
        dw, r, B, P = bsde_solver.bsde.sample(128)
        valid_data = dw, r, B, P
        with tf.GradientTape(persistent=True) as tape:
          loss = bsde_solver.loss_fn(valid_data, training=False)
        grad = tape.gradient(loss, bsde_solver.model.trainable_variables)
        del tape
        #calculate the 2-norm of the gradient after training and make sure it comes to locsl minima
        g_sum = 0
        num = 0
        for g in grad:
            g = g.numpy().flatten()
            g_sum = g_sum + np.sum(g**2)
            num = num + g.shape[0]
        norm = np.sqrt(g_sum/num)
        assert norm <= GRAD_ERROR
        assert np.abs(loss) <= LOSS_ERROR

    # def test_bsde_pipeline(self):
    #     '''
    #     price the single FRA clean value and CVA via the pricing pipeline and compare it with the analytical solution
    #     '''
    #     dim = 1  # dimension of brownian motion
    #     P = 2048  # number of outer Monte Carlo Loops
    #     batch_size = 128
    #     total_time = 1.0
    #     swap_time = 2.0
    #     num_time_interval = 100
    #     sigma = 0.002
    #     kappa = 0.04
    #     theta = 1.0
    #     r_init = 0.03
    #     strike = 0.001
    #     config = {
    #
    #         "eqn_config": {
    #             "_comment": "a basket call option",
    #             "eqn_name": "InterestRateSwap",
    #             "total_time": total_time,
    #             "swap_time": swap_time,
    #             "dim": dim,
    #             "num_time_interval": num_time_interval,
    #             "strike": strike,
    #             "r_init": r_init,
    #             "sigma": sigma,
    #             "kappa": kappa,
    #             "theta": theta,
    #         },
    #
    #         "net_config": {
    #             "y_init_range": [-2.0, 2.0],  # [154.37,165.41], #set to None when not sure
    #             "num_hiddens": [dim + 4, dim + 4],
    #             "lr_values": [5e-2, 5e-3],  # [5e-1,5e-2, 5e-3],
    #             "lr_boundaries": [2000],  # [1000,2000],
    #             "num_iterations": 4000,
    #             "batch_size": batch_size,
    #             "valid_size": 128,
    #             "logging_frequency": 500,
    #             "dtype": "float64",
    #             "verbose": True
    #         }
    #     }
    #     config = munch.munchify(config)
    #     bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    #     tf.keras.backend.set_floatx(config.net_config.dtype)
    #
    #     PRICE_ERROR = 1e-3
    #     bsde_solver = BSDESolver(config, bsde)
    #     training_history = bsde_solver.train()
    #     simulations = bsde_solver.model.simulate_path(bsde.sample(P))
    #     def zcb(t, T):
    #         r = r_init
    #         B = (1 - math.exp(-kappa * (T - t))) / kappa
    #         A = math.exp((B - T + t) * (kappa ** 2 * theta - sigma ** 2 / 2) / kappa ** 2 + \
    #                      (sigma * B) ** 2 / (4 * kappa))
    #         result = A * math.exp(-B * r)
    #         return result
    #     exact = zcb(0, total_time) - zcb(0, swap_time) - (swap_time - total_time) * strike * zcb(0, swap_time) #calculate the analytical solution
    #     print(simulations[0, 0, 0], exact, np.abs((simulations[0, 0, 0] - exact)))
    #     assert np.abs((simulations[0, 0, 0] - exact)) < PRICE_ERROR

if __name__ == "__main__":
    pytest.main()

