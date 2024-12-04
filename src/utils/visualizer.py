"""
visualizer.py
Visualize the prior and posterior distributions as well as the final VaR/ES modelling results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.tsa.stattools import acf


def drawDataset(data: pd.DataFrame):
    """
    Compute and visualize the prior distribution of the log returns
    """
    fig, ax = plt.subplots()

    # Create a histogram of the data
    n, bins, patches = ax.hist(data["log return"], 50, density=True, facecolor='C0', alpha=0.75)
    ax.set_ylabel('Probability mass/density')
    ax.set_xlabel('log return')

    # fit the data with a normal distribution
    mu, std = norm.fit(data["log return"])

    # Generate x values from min to max of your data for plotting the PDF
    xmin, xmax = plt.xlim()  # Get current x-axis limits
    x = np.linspace(xmin, xmax, 100)  # Generate evenly spaced values

    # Calculate the PDF using the fitted parameters
    p = norm.pdf(x, mu, std)

    # Plot the PDF on top of the histogram
    plt.plot(x, p, 'k', linewidth=2)  # 'k' is for black color line
    ax.text(-0.08, 50, f"mu={mu:.6f}, sigma={std:.6f}")

    # Display the plot
    plt.savefig("data/dataset.png")

def checkSamplerConvergence(data: np.ndarray, fig_name: str):
    "Check the convergence of results yield by the Gibbs sampler"
    plt.cla()
    plt.plot(data)
    plt.title(fig_name)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.savefig(f"data/{fig_name}_samples.png")

def checkSamplerACF(data: np.ndarray, fig_name: str):
    "Check the autocovariance of samples generated by the Gibbs sampler, expecting convergence to zero"
    plt.cla()
    plt.title(fig_name)
    mcmc_sample = data

    lags = 40
    acf_values = acf(mcmc_sample, nlags=lags, fft=True)
    
    # Plot the ACF
    plt.stem(range(lags+1), acf_values)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')

    plt.savefig(f"data/{fig_name}_acf.png")
    
    
def drawResults(data: np.ndarray, datatype: str):
    plt.cla()
    fig, ax = plt.subplots()

    # Create a histogram of the data
    n, bins, patches = ax.hist(data, 50, density=True, facecolor='C0', alpha=0.75)
    ax.set_ylabel('Probability mass/density')
    ax.set_xlabel(f'{datatype}')

    # fit the data with a normal distribution
    mu, std = norm.fit(data)

    # Generate x values from min to max of your data for plotting the PDF
    xmin, xmax = plt.xlim()  # Get current x-axis limits
    x = np.linspace(xmin, xmax, 100)  # Generate evenly spaced values

    # Calculate the PDF using the fitted parameters
    p = norm.pdf(x, mu, std)

    # Plot the PDF on top of the histogram
    plt.plot(x, p, 'k', linewidth=2)  # 'k' is for black color line
    if datatype == "VaR":
        ax.text(-0.08, 50, f"mu={mu:.6f}, sigma={std:.6f}")
    elif datatype == "ES":
        ax.text(-0.1, 50, f"mu={mu:.6f}, sigma={std:.6f}")

    # Display the plot
    plt.savefig(f"data/{datatype}.png")