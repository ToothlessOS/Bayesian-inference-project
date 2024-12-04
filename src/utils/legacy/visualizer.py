import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

"""
Pass in a pandas dataframe of desingated format to create visualization
"""
def draw(df: pd.DataFrame, type: str = 'closed'):
    
    fig, ax = plt.subplots()

    # the histogram of the data
    n, bins, patches = ax.hist(df["涨跌幅"], 50, density=True, facecolor='C0', alpha=0.75)
    ax.set_ylabel('Probability')
    ax.set_xlabel('Closed (in percentage)')
    
    # fit the data with a normal distribution
    mu, std = norm.fit(df["涨跌幅"])
    
    # Generate x values from min to max of your data for plotting the PDF
    xmin, xmax = plt.xlim()  # Get current x-axis limits
    x = np.linspace(xmin, xmax, 100)  # Generate evenly spaced values

    # Calculate the PDF using the fitted parameters
    p = norm.pdf(x, mu, std)

    # Plot the PDF on top of the histogram
    plt.plot(x, p, 'k', linewidth=2)  # 'k' is for black color line
    ax.text(-10, 0.25, f"mu={mu:.4f}, sigma={std:.4f}")

    # Display the plot
    plt.show()