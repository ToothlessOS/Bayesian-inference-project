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

2. Prompt: I have a prior distribution which is a normal with known mean and variance. I also have some additional data points. How should I sample the posterior distribution's mean and variance with Gibbs sampler?

> To sample the posterior distribution's mean ($$\mu$$) and variance ($$\sigma^2$$) using the Gibbs sampler, you need to iteratively sample from their full conditional distributions. Here's how you can approach this step-by-step:
> 
> ---
> 
> ## **Problem Setup**
> 
> You have:
> 
> - A prior distribution for $$\mu$$ (mean) and $$\sigma^2$$ (variance).
> - Observed data points $$y_1, y_2, \dots, y_n$$.
> 
> Assume:
> 
> 1. The prior for $$\mu$$ is $$N(\mu_0, \tau_0^2)$$.
> 2. The prior for $$\sigma^2$$ is an inverse-gamma distribution: $$\sigma^2 \sim IG(a_0, b_0)$$.
> 3. The likelihood is $$y_i \sim N(\mu, \sigma^2)$$.
> 
> The posterior distribution for $$(\mu, \sigma^2)$$ combines the prior and likelihood.
> 
> ---
> 
> ## **Full Conditional Distributions**
> 
> 1. **Conditional distribution of $$\mu | \sigma^2, y$$:**
>    
>    - Given the data and current value of $$\sigma^2$$, the conditional posterior of $$\mu$$ is normal:
>      
>      $$
>      \mu | \sigma^2, y \sim N\left( \mu_n, \tau_n^2 \right),
>      $$
>      
>      where:
>    
>    - $$ \mu_n = \frac{\frac{\mu_0}{\tau_0^2} + \frac{n\bar{y}}{\sigma^2}}{\frac{1}{\tau_0^2} + \frac{n}{\sigma^2}} $$,
>    
>    - $$ \tau_n^2 = \left( \frac{1}{\tau_0^2} + \frac{n}{\sigma^2} \right)^{-1} $$,
>    
>    - $$ n $$ is the number of data points,
>    
>    - $$ \bar{y} = \frac{1}{n} \sum_{i=1}^n y_i $$ is the sample mean.
> 
> 2. **Conditional distribution of $$\sigma^2 | \mu, y$$:**
>    
>    - Given the data and current value of $$\mu$$, the conditional posterior of $$\sigma^2$$ is inverse-gamma:
>      
>      $$
>      \sigma^2 | \mu, y \sim IG(a_n, b_n),
>      $$
>      
>      where:
>    
>    - $$ a_n = a_0 + n/2 $$,
>    
>    - $$ b_n = b_0 + 0.5\sum_{i=1}^n (y_i - \mu)^2 $$.
> 
> ---
> 
> ## **Gibbs Sampling Algorithm**
> 
> The Gibbs sampler alternates between sampling from these two conditional distributions:
> 
> 1. Initialize starting values for both parameters, e.g., set:
>    
>    - $$ \mu^{(0)} = 0,\, (\sigma^2)^{(0)} = 1.$$
> 
> 2. For each iteration $$t = 1, 2, ..., T$$:
>    
>    - Sample $$ \mu^{(t)} \sim N(\mu_n, \tau_n^2) $$, using the current value of $$ (\sigma^2)^{(t-1)}.$$
>    - Sample $$ (\sigma^2)^{(t)} \sim IG(a_n, b_n) ,$$ using the current value of $$ \mu^{(t)}.$$
> 
> 3. Repeat for a sufficient number of iterations to ensure convergence.
> 
> 4. Discard initial "burn-in" samples and use the remaining samples to approximate the posterior distributions.
> 
> ---
> 
> ## **Implementation in R**
> 
> Here’s an example implementation:
> 
> ```R
> # Data and priors
> y <- c(10.5, 11.3, 9.8, 10.9) # Example data
> n <- length(y)
> y_bar <- mean(y)
> s_squared <- var(y)
> 
> # Prior parameters
> mu_0 <- 10
> tau_0_squared <- 4
> a_0 <- 3
> b_0 <- 10
> 
> # Gibbs sampler settings
> iterations <- 10000
> burn_in <- 1000
> 
> # Initialize storage
> mu_samples <- numeric(iterations)
> sigma_squared_samples <- numeric(iterations)
> 
> # Initial values
> mu_samples[1] <- mu_0
> sigma_squared_samples[1] <- b_0 / (a_0 - 1)
> 
> # Gibbs sampling loop
> for (t in 2:iterations) {
>   # Sample mu given sigma_squared
>   sigma_squared_t_minus_1 <- sigma_squared_samples[t-1]
>   tau_n_squared <- 1 / (1/tau_0_squared + n/sigma_squared_t_minus_1)
>   mu_n <- tau_n_squared * (mu_0/tau_0_squared + n*y_bar/sigma_squared_t_minus_1)
>   mu_samples[t] <- rnorm(1, mean = mu_n, sd = sqrt(tau_n_squared))
> 
>   # Sample sigma_squared given mu
>   mu_t <- mu_samples[t]
>   a_n <- a_0 + n/2
>   b_n <- b_0 + 0.5 * sum((y - mu_t)^2)
>   sigma_squared_samples[t] <- 1 / rgamma(1, shape = a_n, rate = b_n)
> }
> 
> # Remove burn-in samples
> mu_samples <- mu_samples[-(1:burn_in)]
> sigma_squared_samples <- sigma_squared_samples[-(1:burn_in)]
> 
> # Summarize results
> cat("Posterior mean of mu:", mean(mu_samples), "\n")
> cat("Posterior mean of sigma^2:", mean(sigma_squared_samples), "\n")
> ```
> 
> ---
> 
> ## **Key Points**
> 
> - The Gibbs sampler relies on cycling through the full conditional distributions.
> - Convergence diagnostics are essential to ensure that the chain has reached its stationary distribution.
> - The posterior samples can be used to compute summaries (e.g., mean or credible intervals).
> 
> This approach effectively approximates the posterior distribution of both parameters given your prior beliefs and observed data.
> 
> Citations:
> [1] https://bookdown.org/rdpeng/advstatcomp/gibbs-sampler.html
> [2] https://cja.education.illinois.edu/docs/librariesprovider37/edpsy590ca/lectures/5-mcmc-gibbs/mcmc_gibbs_sampling-post.pdf?Status=Master&sfvrsn=56bf73e8_5
> [3] https://malouche.github.io/BayesianStatistics/Gibbs_sampler.html
> [4] https://www.y1zhou.com/series/bayesian-stat/bayesian-stat-normal-two-param-setting/
> [5] https://en.wikipedia.org/wiki/Gibbs_sampling
> [6] http://optimumsportsperformance.com/blog/bayesian-simple-linear-regression-by-hand-gibbs-sampler/

## 5. Definitions

1. Conjugate prior: A conjugate prior is a type of prior distribution in Bayesian statistics that, when combined with a likelihood function, results in a posterior distribution belonging to the same family as the prior distribution. This property simplifies Bayesian inference by allowing for analytical solutions when deriving posterior distributions, rather than requiring numerical methods.

## 6. Useful resources

1. [On Bayesian inference of normal distributions](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/lectures/lecture5.pdf) - The documented version of the info given in class
