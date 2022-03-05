import numpy as np
import tensorflow as tf
from tqdm import tqdm
import math
from scipy.stats import multivariate_normal as normal


class FiccRecursiveEquation(object):
    """Base class for defining PDE related function."""

    def __init__(self, eqn_config):
        self.dim = eqn_config.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.y_init = None
        #self.clean_bsde = eqn_config.clean_bsde
        #self.clean_bsde_model = eqn_config.clean_bsde_model

    def sample(self, num_sample):
        """Sample forward SDE."""
        """Sample clean BSDE and the collateral account."""
        raise NotImplementedError

    def f_tf(self, t, r, x, y, z, vhat, collateral):
        """Generator function in the PDE."""
        raise NotImplementedError

    def g_tf(self, t, x, vhat,collateral):
        """Terminal condition of the PDE."""
        raise NotImplementedError


class InterestRateSwapCVA(FiccRecursiveEquation):
    def __init__(self, eqn_config):
        super(InterestRateSwapCVA, self).__init__(eqn_config)
        self.strike = eqn_config.strike # strike of fix leg
        self.r_0 = eqn_config.r_init
        self.r_init = np.ones(self.dim) * eqn_config.r_init  # initial value of r, the underlying
        self.sigma = eqn_config.sigma
        self.kappa = eqn_config.kappa
        self.theta = eqn_config.theta
        self._recoveryC = eqn_config.recoveryC
        self._lambdaC = eqn_config.lambdaC
        self._recoveryB = eqn_config.recoveryB
        self._lambdaB = eqn_config.lambdaB
        self.clean_value = eqn_config.clean_value  # Class Equation, to simulate dw and x
        self.clean_value_model = eqn_config.clean_value_model

        self.swap_time = eqn_config.swap_time
        self.delta_T = self.swap_time - self.total_time
        self.zero_curve = np.array([self.discount_curve(0,i * self.sqrt_delta_t) for i in range(self.num_time_interval + 1)])
        #self.r = eqn_config.r
        #self.useExplict = False  # whether to use explict formula to evaluate dyanamics of x

    def discount_curve(self, t, T):
        '''
        The analytical formula to calculate the zero coupon bond price give the short rate r
        '''
        kappa = self.kappa
        theta = self.theta
        sigma = self.sigma
        r = self.r_0
        #t = self.t
        #T = self.T
        B = (1 - math.exp(-kappa * (T - t))) / kappa
        A = math.exp((B - T + t)*(kappa**2 * theta - sigma**2/2)/kappa**2 +\
                     (sigma*B)**2/(4*kappa))
        result = A * math.exp(-B * r)
        return result

    def SIGMA(self,t,T):
        alpha = self.kappa
        B = (1 - math.exp(-alpha * (T - t))) / alpha
        return self.sigma*B

    def sample(self, num_sample): #sample r_t, P_t (denote as x), B_t
        forward_sde = self.clean_value.sample(num_sample)
        dw, r, B, P = forward_sde
        clean_value = self.clean_value_model.simulate_path(forward_sde)

        # this is where we model the collateral account.
        collateral = clean_value * 0.0
        return dw, r, B, P, clean_value, collateral

    def f_tf(self, t, r, p, y, z, v_clean, collateral):
        '''
        Generator function of the BSDE of cva and dva
        '''
        cva = (1.0 - self._recoveryC) * tf.maximum(-v_clean + collateral, 0.0) * self._lambdaC
        dva = (1.0 - self._recoveryB) * tf.maximum(v_clean - collateral, 0.0) * self._lambdaB
        discount = (r + self._lambdaC + self._lambdaB) * y
        return -cva + dva - discount

    def g_tf(self, t, p, vhat, collateral):
        '''
        the terminal payoff of cva is 0
        '''
        return 0

    def monte_carlo(self, num_sample=1024): #monte carlo estimation of CVA and DVA
        '''
        implement algorithm 2, calculate cva and dva based on the pricing formula via monte carlo given the clean value process of V
        return mean and confidence interval of simulation
        '''
        estimate = []
        for i in tqdm(range(n)):  # split into batches to estimate
            dw, r, B, P, clean_value, collateral = self.sample(1024)  # shape (num_sample, dim=1, num_time_interval+1)
            discount = (1/B) * np.exp(-(self._lambdaB+self._lambdaC)*np.linspace(0,self.total_time,self.num_time_interval+1))
            phi_cva = (1 - self._recoveryC) * discount * np.maximum(collateral - clean_value, 0) * self._lambdaC
            phi_dva = (1 - self._recoveryB) * discount * np.maximum(clean_value - collateral, 0) * self._lambdaB

            # trapeziodal rule
            cva = np.sum(phi_cva, axis=-1) - (phi_cva[:, :, -1] + phi_cva[:, :, 0]) / 2
            dva = np.sum(phi_dva, axis=-1) - (phi_dva[:, :, -1] + phi_dva[:, :, 0]) / 2

            estimate += list(dva[:, 0] - cva[:, 0])

        if num_sample % 1024 != 0:  # calculate the remaining number of smaples
            dw, r, B, P, clean_value, collateral = self.sample(num_sample % 1024)  # shape (num_sample, dim=1, num_time_interval+1)
            discount = (1 / B) * np.exp(-(self._lambdaB + self._lambdaC) * np.linspace(0, self.total_time, self.num_time_interval + 1))
            phi_cva = (1 - self._recoveryC) * discount * np.maximum(collateral - clean_value, 0) * self._lambdaC
            phi_dva = (1 - self._recoveryB) * discount * np.maximum(clean_value - collateral, 0) * self._lambdaB

            # trapeziodal rule
            cva = np.sum(phi_cva, axis=-1) - (phi_cva[:, :, -1] + phi_cva[:, :, 0]) / 2
            dva = np.sum(phi_dva, axis=-1) - (phi_dva[:, :, -1] + phi_dva[:, :, 0]) / 2

            estimate += list(dva[:, 0] - cva[:, 0])
        n = num_sample//1024
        estimate = np.array(estimate) * self.delta_t  # times time-interval (height of a trapezium)
        mean = np.mean(estimate)
        std = np.std(estimate) / np.sqrt(n)

        return mean, [mean - 3 * std, mean + 3 * std]
