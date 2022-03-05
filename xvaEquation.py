from equation import Equation, FICC_Equation
import numpy as np
import tensorflow as tf
import math
from scipy.stats import multivariate_normal as normal
from scipy.stats import norm

class InterestRateSwap(FICC_Equation):
    '''
    the class of the BSDE of clean value of the product
    '''
    def __init__(self, eqn_config):
        super(InterestRateSwap, self).__init__(eqn_config)
        self.strike = eqn_config.strike # strike of fix leg
        self.r_0 = eqn_config.r_init
        self.r_init = np.ones(self.dim) * eqn_config.r_init  # initial value of r, the underlying
        self.sigma = eqn_config.sigma
        self.kappa = eqn_config.kappa
        self.theta = eqn_config.theta
        #self.principal = eqn_config.principal
        self.swap_time = eqn_config.swap_time
        self.delta_T = self.swap_time - self.total_time
        # self.zero_curve = np.array([self.discount_curve(0,i * self.sqrt_delta_t) for i in range(self.num_time_interval + 1)])

    def discount_curve(self, r, t, T): #return a tensor (num_sample, dim, num_time_interval + 1)
        '''
        The analytical formula to calculate the zero coupon bond price give the short rate r
        '''
        kappa = self.kappa
        theta = self.theta
        sigma = self.sigma
        B = (1 - math.exp(-kappa * (T - t))) / kappa
        A = math.exp((B - T + t)*(kappa**2 * theta - sigma**2/2)/kappa**2 +\
                     (sigma*B)**2/(4*kappa))
        result = A * np.exp(-B * r)
        return result

    def SIGMA(self,t,T):
        alpha = self.kappa
        B = (1 - math.exp(-alpha * (T - t))) / alpha
        return self.sigma*B

    def sample(self, num_sample): #sample r_t, P_t (denote as x), B_t
        '''
        sample increment of brownian motion, short rate, money market account and zero coupon bond price
        '''
        #sample increment of Brownian motion
        dw_sample = normal.rvs(size=[num_sample,
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        if self.dim == 1:
            dw_sample = np.expand_dims(dw_sample, axis=0)
            dw_sample = np.swapaxes(dw_sample, 0, 1)
        #simulate short rate r_t
        r_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        r_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.r_init
        for i in range(self.num_time_interval):
            r_sample[:, :, i + 1] = (1 - self.kappa*self.delta_t)*r_sample[:, :, i] + self.kappa*self.theta*self.delta_t + \
                                    (self.sigma  * dw_sample[:, :, i])

        B_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        B_sample[:, :, 0] = np.ones([num_sample, self.dim])

        Phat_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        Phat_sample[:, :, 0] = self.discount_curve(np.ones([num_sample, self.dim]) * self.r_init, 0.0, self.swap_time)
        
        #sample the process of P/B
        for i in range(self.num_time_interval):
            Phat_sample[:, :, i+1] = Phat_sample[:, :, i] * np.exp(-(self.SIGMA(i*self.delta_t,self.swap_time)**2)*self.delta_t/2 + \
                       self.SIGMA(i*self.delta_t,self.swap_time)* dw_sample[:, :, i])
        #P_sample = Phat_sample * B_sample
        P_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        P_sample[:, :, 0] = self.discount_curve(np.ones([num_sample, self.dim]) * self.r_init, 0.0, self.swap_time)
        
        #sample the process of P
        for i in range(self.num_time_interval):
            P_sample[:, :, i + 1] = self.discount_curve(r_sample[:, :, i + 1], (i + 1)*self.delta_t, self.swap_time)
        B_sample = P_sample/Phat_sample

        return dw_sample, r_sample, B_sample, P_sample

    def f_tf(self, t, r, p, y, z):
        return -r * y

    def g_tf(self, t, p):
        return 1 - (1 + self.delta_T * self.strike) * p

