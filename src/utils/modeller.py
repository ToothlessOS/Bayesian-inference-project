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
                   tau_prior_alpha: int, tau_prior_beta: int, mu_prior_mean: int, mu_prior_tau: int, # The priors req'd
                   test = False, tau_init = 0, mean_init = 0): # For testing the convergence of the sampler / determine burn-in
    # y: the observed data
    # Intialize the values for gibbs sampler
    # Here we use the mean of the prior
    tau = tau_prior_alpha * tau_prior_beta
    mean = mu_prior_mean

    tau_samples = np.zeros(n)
    mean_samples = np.zeros(n)
    if test == False:
        # Two arrays to store the sampled values
        tau_samples[0] = tau
        mean_samples[0] = mean
    else:
        tau_samples[0] = tau_init
        mean_samples[0] = mean_init    

    for i in range(1, n):
        
        gamma_sampler.reset(tau_prior_alpha + len(y)/2, tau_prior_beta + np.sum((y - mean_samples[i-1])**2)/2)
        tau_given_mean_and_data = gamma_sampler.sample() # The first full conditional distribution req'd
        tau_samples[i] = tau_given_mean_and_data
        
        normal_sampler.reset(tau_samples[i-1]*(np.sum(y)+mu_prior_mean*mu_prior_tau)/(len(y)*tau_samples[i-1] + mu_prior_tau), 1/(len(y)*tau_samples[i-1] + mu_prior_tau))
        mean_given_tau_and_data = normal_sampler.sample() # The second full conditional distribution req'd
        mean_samples[i] = mean_given_tau_and_data

    return mean_samples, tau_samples
        
def get_VaR(mean_samples: np.ndarray, tau_samples: np.ndarray, confidence = 0.95):

    VaR_array = np.zeros(len(mean_samples))
    for i in range(len(mean_samples)):
        VaR = norm.ppf(1 - confidence, loc=mean_samples[i], scale=(1/(tau_samples[i]))**0.5) #Compute VaR
        VaR_array[i] = VaR

    return VaR_array
    

def get_ES(mean_samples: np.ndarray, tau_samples: np.ndarray, confidence = 0.95):
    
    ES_array = np.zeros(len(mean_samples))
    for i in range(len(mean_samples)):
        z_alpha = norm.ppf(confidence)
        phi_z_alpha = norm.pdf(z_alpha)
        ES = mean_samples[i] - (1/(tau_samples[i]))**0.5 * phi_z_alpha / (1 - confidence)
        ES_array[i] = ES

    return ES_array