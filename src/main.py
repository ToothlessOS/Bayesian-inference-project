import utils.dataloader as dataloader
import utils.modeller as modeller
import datetime
import random
import utils.visualizer as visualizer

# Load and cleanup data
data = dataloader.get("sh000001", datetime.date(2021, 1, 1), datetime.date(2024, 11, 24))
print(data)
visualizer.drawDataset(data)

# Set up the prior distribution and additional data and pass into gibbs sampler
y = data['log return'].to_numpy()

# Check convergence for different initial values / determine the burn-in
mu_samples_collection = []
tau_samples_collection = []
mu_init_collection = []
tau_init_collection = []
for i in range(5):
    _mu_init = random.uniform(-1, 1)
    _tau_init = random.uniform(0, 1000)
    _mu_samples, _tau_samples = modeller.gibbs_sampling(200, modeller.gamma_sampler(0,0), modeller.normal_sampler(0,0),
                                                        y,
                                                        1, 0.001, 0, 0.001,
                                                        True, _tau_init, _mu_init)
    mu_samples_collection.append(_mu_samples)
    tau_samples_collection.append(_tau_samples)
    mu_init_collection.append(_mu_init)
    tau_init_collection.append(_tau_init)

visualizer.determineBurnIn(mu_samples_collection, tau_samples_collection, mu_init_collection, tau_init_collection, "burn-in check")

# Run the gibbs sampler
mean_samples, tau_samples = modeller.gibbs_sampling(10000, modeller.gamma_sampler(0,0), modeller.normal_sampler(0,0),
                                                    y, # Additional data
                                                    1, 0.001, 0, 0.001) # Prior parameters

# Visualize the results of the gibbs sampler
visualizer.checkSamplerConvergence(mean_samples, "mean")
visualizer.checkSamplerConvergence(tau_samples, "tau")

# Setup the apprioriate burn-in params here
burn_in = 100
mean_samples = mean_samples[burn_in:]
tau_samples = tau_samples[burn_in:]

visualizer.checkSamplerACF(mean_samples, "mean ACF")
visualizer.checkSamplerACF(tau_samples, "tau ACF")

# Calculate and plot VaR
VaR_samples = modeller.get_VaR(mean_samples, tau_samples, 0.95)
visualizer.drawResults(VaR_samples, "VaR")

# Calculate and plot ES
ES_samples = modeller.get_ES(mean_samples, tau_samples, 0.95)
visualizer.drawResults(ES_samples, "ES")

