import utils.dataloader as dataloader
import utils.modeller as modeller
import datetime
import utils.visualizer as visualizer

# Load and cleanup data
data = dataloader.get("sh000001", datetime.date(2021, 1, 1), datetime.date(2024, 11, 24))
print(data)
visualizer.drawDataset(data)

# Set up the prior distribution and additional data and pass into gibbs sampler
y = data['log return'].to_numpy()
mean_samples, tau_samples = modeller.gibbs_sampling(10000, modeller.gamma_sampler(0,0), modeller.normal_sampler(0,0),
                                                    y, # Additional data
                                                    1, 0.001, 0, 0.001) # Prior parameters

# Visualize the results of the gibbs sampler
visualizer.checkSamplerConvergence(mean_samples, "mean")
visualizer.checkSamplerConvergence(tau_samples, "tau")

# Pick the from the 100th sample for convergence
mean_samples = mean_samples[1000:]
tau_samples = tau_samples[1000:]

visualizer.checkSamplerACF(mean_samples, "mean ACF")
visualizer.checkSamplerACF(tau_samples, "tau ACF")

# Calculate and plot VaR
VaR_samples = modeller.get_VaR(mean_samples, tau_samples, 0.95)
visualizer.drawResults(VaR_samples, "VaR")

# Calculate and plot ES
ES_samples = modeller.get_ES(mean_samples, tau_samples, 0.95)
visualizer.drawResults(ES_samples, "ES")

