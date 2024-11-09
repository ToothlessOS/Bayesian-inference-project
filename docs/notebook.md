# Notebook for implementation

## 1. Value-at-Risk (VaR)

[金融学习笔记（十二）：VaR(Value at Risk) - 知乎](https://zhuanlan.zhihu.com/p/412026199)

## 2. Expected Shortfall (ES)

[什么是 Expected Shortfall (ES),相比 VaR 它有什么优点？ - 知乎](https://www.zhihu.com/question/51078169?from=profile_question_card)

It is obvious that the most important thing we need here is the **posterior distribution** for computing both VaR and ES.

## 3. Procedure

1. Priori: directly take previous index data and model them with a normal distribution

2. Posterior / Likelihood: using Bayesian inference and Gibbs sampler

## 4. Modelling suggestions from GPT

1. Prompt: I have a prior distribution which has already been modelled with a normal distribution with both known mean and known variance. Now I have collected some additional data points, which I assume to be following a normal distribution with unknown mean and unknown variance. How should I perform a Bayesian inference to find the estimated posterior distribution? Please give me step-by-step guidance with ample reasoning. Thanks!

> To perform Bayesian inference in this scenario, we'll follow a step-by-step process to estimate the posterior distribution. This is a case of conjugate Bayesian inference for a normal distribution with unknown mean and unknown variance.
> 
> ## Step 1: Define the Prior Distribution
> 
> Your prior distribution for the mean (μ) is already modeled as a normal distribution:
> 
> μ ~ N(μ₀, σ₀²)
> 
> Where μ₀ is the known prior mean and σ₀² is the known prior variance.
> 
> ## Step 2: Define the Likelihood
> 
> The likelihood of your data, assuming it follows a normal distribution with unknown mean (μ) and unknown variance (σ²), is:
> 
> X₁, X₂, ..., Xₙ ~ N(μ, σ²)
> 
> ## Step 3: Choose a Prior for the Variance
> 
> Since the variance is also unknown, we need to specify a prior for it. The conjugate prior for the variance of a normal distribution is the Inverse-Gamma distribution. Let's denote it as:
> 
> σ² ~ InvGamma(α, β)
> 
> Where α and β are hyperparameters that need to be specified based on your prior beliefs about the variance.
> 
> ## Step 4: Calculate Summary Statistics
> 
> Calculate the sample mean (x̄) and sample variance (s²) from your new data:
> 
> x̄ = (1/n) Σ Xᵢ
> s² = (1/(n-1)) Σ (Xᵢ - x̄)²
> 
> ## Step 5: Derive the Posterior Distribution
> 
> The posterior distribution for the mean and variance will be a Normal-Inverse-Gamma distribution:
> 
> (μ, σ²) | data ~ NIG(μₙ, κₙ, αₙ, βₙ)
> 
> Where:
> 
> κₙ = κ₀ + n
> μₙ = (κ₀μ₀ + nx̄) / κₙ
> αₙ = α₀ + n/2
> βₙ = β₀ + 0.5Σ(xᵢ - x̄)² + (κ₀n(x̄ - μ₀)²) / (2κₙ)
> 
> Here, κ₀ = 1/σ₀² (the precision of the prior for μ).
> 
> ## Step 6: Interpret the Posterior Distribution
> 
> The marginal posterior distribution for μ is a t-distribution:
> 
> μ | data ~ t₂αₙ(μₙ, βₙ/(αₙκₙ))
> 
> And the marginal posterior distribution for σ² is an Inverse-Gamma distribution:
> 
> σ² | data ~ InvGamma(αₙ, βₙ)
> 
> ## Step 7: Calculate Point Estimates and Credible Intervals
> 
> For the mean (μ):
> 
> - Point estimate: μₙ
> - 95% Credible Interval: μₙ ± t₀.₀₂₅(2αₙ) * √(βₙ/(αₙκₙ))
> 
> For the variance (σ²):
> 
> - Point estimate: βₙ / (αₙ - 1) (for αₙ > 1)
> - 95% Credible Interval: [1/qgamma(0.975, αₙ, βₙ), 1/qgamma(0.025, αₙ, βₙ)]
> 
> ## Reasoning
> 
> 1. We use conjugate priors (Normal for μ, Inverse-Gamma for σ²) because they lead to closed-form solutions for the posterior, making calculations more tractable[1][2].
> 
> 2. The Normal-Inverse-Gamma distribution is the conjugate prior for a normal likelihood with unknown mean and variance, which is why our posterior follows this distribution[4].
> 
> 3. The t-distribution for the marginal posterior of μ accounts for the uncertainty in both μ and σ², providing a more robust estimate than a normal distribution would[4].
> 
> 4. The Inverse-Gamma distribution for the marginal posterior of σ² reflects the updated beliefs about the variance after observing the data[4].
> 
> 5. Point estimates and credible intervals provide practical summaries of the posterior distributions, allowing for straightforward interpretation of the results[4].
> 
> This approach provides a full Bayesian treatment of the problem, accounting for uncertainties in both the mean and variance of the normal distribution. It allows you to update your prior beliefs about these parameters based on the new data you've collected, resulting in a posterior distribution that combines your prior knowledge with the information from the data.
> 
> Citations:
> [1] https://en.wikipedia.org/wiki/Bayesian_inference
> [2] https://mcs.utm.utoronto.ca/~nosedal/sta313/sta313-normal-mean.pdf
> [3] https://tinyheero.github.io/2017/03/08/how-to-bayesian-infer-101.html
> [4] https://www.statlect.com/fundamentals-of-statistics/normal-distribution-Bayesian-estimation
> [5] https://www.youtube.com/watch?v=c-d05z0_5mw
> [6] https://stephens999.github.io/fiveMinuteStats/shiny_normal_example.html
> [7] https://www.youtube.com/watch?v=MIAm7eUfMok

2. Prompt: How would MCMC sampling techniques like MCMC be used in this modelling process?

> MCMC (Markov Chain Monte Carlo) sampling techniques can be very useful in the Bayesian inference process you described, especially when dealing with complex posterior distributions that are not analytically tractable. Here's how MCMC, particularly the Metropolis algorithm, could be applied to your scenario:
> 
> ## Why Use MCMC?
> 
> In your case, while you have a conjugate prior (normal distribution) for the mean of a normal distribution, the addition of an unknown variance complicates the posterior. The resulting posterior distribution is a Normal-Inverse-Gamma, which can be challenging to work with directly, especially if you need to calculate expectations or credible intervals[1].
> 
> ## Applying MCMC to Your Model
> 
> 1. **Define the Target Distribution**
> 
> The target distribution is your posterior distribution, which is proportional to:
> 
> P(μ, σ² | data) ∝ P(data | μ, σ²) * P(μ) * P(σ²)
> 
> 2. **Initialize the Markov Chain**
> 
> Start with initial guesses for μ and σ². For example:
> 
> - μ₀ = sample mean of your data
> 
> - σ²₀ = sample variance of your data
> 3. **Propose New Values**
> 
> At each iteration, propose new values for μ and σ² using a proposal distribution. A common choice is a normal distribution centered at the current values:
> 
> μ_new ~ N(μ_current, step_size_μ)
> σ²_new ~ N(σ²_current, step_size_σ²)
> 
> Ensure σ²_new > 0 by taking its absolute value or using a log-normal proposal.
> 
> 4. **Calculate Acceptance Ratio**
> 
> Compute the ratio:
> 
> r = [P(data | μ_new, σ²_new) * P(μ_new) * P(σ²_new)] / [P(data | μ_current, σ²_current) * P(μ_current) * P(σ²_current)]
> 
> 5. **Accept or Reject**
> 
> Accept the new values with probability min(1, r). If accepted, move to (μ_new, σ²_new); otherwise, stay at (μ_current, σ²_current).
> 
> 6. **Repeat and Sample**
> 
> Repeat steps 3-5 for many iterations. After a "burn-in" period, start collecting samples.
> 
> ## Benefits of MCMC in This Context
> 
> 1. **Flexibility**: MCMC can handle the joint posterior of μ and σ² without needing to derive the marginal distributions analytically[2].
> 
> 2. **Correlation Capture**: The samples will naturally capture any correlation between μ and σ² in the posterior.
> 
> 3. **Easy Inference**: Once you have samples, it's straightforward to compute summary statistics, credible intervals, or any other quantity of interest.
> 
> 4. **Adaptability**: If you later want to change your model (e.g., add more parameters or change the prior), MCMC methods can often be easily adapted.
> 
> ## Implementation Considerations
> 
> 1. **Tuning**: The step sizes in the proposal distributions need to be tuned for efficient sampling. Too large, and most proposals will be rejected; too small, and the chain will move slowly[3].
> 
> 2. **Convergence**: Use multiple chains and convergence diagnostics (e.g., Gelman-Rubin statistic) to ensure your samples are representative of the true posterior.
> 
> 3. **Burn-in and Thinning**: Discard the initial samples (burn-in) and potentially keep only every nth sample (thinning) to reduce autocorrelation.
> 
> 4. **Software**: Consider using established MCMC software like PyMC, Stan, or JAGS, which handle many implementation details automatically[4].
> 
> By using MCMC, you can obtain a set of samples that represent your posterior distribution. From these samples, you can easily compute means, credible intervals, and other quantities of interest, providing a full Bayesian treatment of your inference problem.
> 
> Citations:
> [1] https://link.springer.com/article/10.3758/s13423-016-1015-8
> [2] https://en.wikipedia.org/wiki/Markov_chain_monte_carlo
> [3] https://towardsdatascience.com/bayesian-inference-problem-mcmc-and-variational-inference-25a8aa9bce29?gi=b9f8ae748bb2
> [4] https://www.quantstart.com/articles/Markov-Chain-Monte-Carlo-for-Bayesian-Inference-The-Metropolis-Algorithm/


