"""
Modeller.py
Implementation of essentials tools for modelling Bayesian inference
"""

import numpy as np
from scipy.stats import norm
from scipy.stats import gamma


class sampler:
    pass

class gamma_sampler(sampler):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample(self):
        return gamma.rvs(self.alpha, self.beta, size=1)[0]
    
    def reset(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

class normal_sampler(sampler):
    def __init__(self, mean, std_dev):
        self.mean = mean
        self.std_dev = std_dev

    def sample(self):
        return norm.rvs(loc=self.mean, scale=self.std_dev, size=1)[0]
    
    def reset(self, mean, std_dev):
        self.mean = mean
        self.std_dev = std_dev

def gibbs_sampling(n: int, # sample size
                   gamma_sampler: gamma_sampler, normal_sampler: normal_sampler, # gamma and normal samplers req'd
                   y: np.ndarray, # additionally observed data
                   tau_prior_alpha: int, tau_prior_beta: int, mu_prior_mean: int, mu_prior_tau: int): # The priors req'd
    # y: the observed data
    # Intialize the values for gibbs sampler    
    mean = 0
    # The burn in value set for convergence
    burn_in = 1000
    # Two arrays to store the sampled values
    tau_samples = np.zeros(n + burn_in)
    mean_samples = np.zeros(n + burn_in)
    for i in range(0, n + burn_in - 1):
        
        gamma_sampler.reset()
        tau_given_mean_and_data = gamma_sampler.sample() # The first full conditional distribution req'd
        tau_samples[i] = tau_given_mean_and_data
        
        normal_sampler.reset()
        tau = tau_samples[i]
        mean_given_tau_and_data = normal_sampler.sample() # The second full conditional distribution req'd
        mean_samples[i+1] = mean_given_tau_and_data

    return mean_samples[burn_in:], tau_samples[burn_in:]
        
def get_VaR(points: np.ndarray, confidence = 0.95):

    z_alpha = np.percentile(points, 100 * (1 - confidence))
    return z_alpha
    

def get_ES(points: np.ndarray, confidence = 0.95):
    
    pass