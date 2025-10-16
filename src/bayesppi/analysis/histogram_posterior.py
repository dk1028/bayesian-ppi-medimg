import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import seaborn as sns

# True parameter values and target quantity
true_theta_A = 0.6
true_theta_H_A1 = 0.8
true_theta_H_A0 = 0.3
true_g = true_theta_A * true_theta_H_A1 + (1 - true_theta_A) * true_theta_H_A0

# Simulation settings
M = 50  # number of simulated data sets
N_A = 1000
N_H = 100

# Storage for summary statistics
means = np.zeros(M)
medians = np.zeros(M)
ci_lower = np.zeros(M)
ci_upper = np.zeros(M)

# Loop over M simulated data sets
for i in range(M):
    # 1) Generate a new synthetic data set
    A_data = np.random.binomial(1, true_theta_A, size=N_A)
    A_H = A_data[:N_H]
    H_data = np.array([
        np.random.binomial(1, true_theta_H_A1 if a == 1 else true_theta_H_A0)
        for a in A_H
    ])

    # 2) Fit the Bayesian chain-rule model
    with pm.Model() as model:
        theta_A = pm.Beta("theta_A", 1, 1)
        theta_H_A1 = pm.Beta("theta_H_A1", 1, 1)
        theta_H_A0 = pm.Beta("theta_H_A0", 1, 1)

        pm.Bernoulli("A_obs", p=theta_A, observed=A_data)
        pm.Bernoulli("H_obs",
                     p=theta_H_A1 * A_H + theta_H_A0 * (1 - A_H),
                     observed=H_data)

        g = pm.Deterministic("g",
                             theta_H_A1 * theta_A +
                             theta_H_A0 * (1 - theta_A))

        trace = pm.sample(draws=500, tune=500,
                          chains=2, cores=1,
                          progressbar=False,
                          return_inferencedata=True)

    # 3) Extract posterior draws of g and compute summary stats
    g_samples = trace.posterior["g"].values.flatten()
    means[i] = g_samples.mean()
    medians[i] = np.median(g_samples)
    ci_lower[i], ci_upper[i] = np.percentile(g_samples, [2.5, 97.5])

# 4) Compute empirical coverage of the 95% credible interval
coverage = np.mean((ci_lower <= true_g) & (true_g <= ci_upper))
print(f"Empirical coverage of 95% CI: {coverage*100:.1f}%")

# 5) Plot histograms of posterior means and medians across simulations
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
sns.histplot(means, bins=15, kde=True, color="steelblue")
plt.axvline(true_g, color="red", linestyle="--", label=f"True g = {true_g:.2f}")
plt.title("Histogram of Posterior Means of g")
plt.xlabel("Posterior mean")
plt.ylabel("Frequency")
plt.legend()
plt.show()

plt.subplot(1, 2, 2)
sns.histplot(medians, bins=15, kde=True, color="seagreen")
plt.axvline(true_g, color="red", linestyle="--", label=f"True g = {true_g:.2f}")
plt.title("Histogram of Posterior Medians of g")
plt.xlabel("Posterior median")
plt.ylabel("Frequency")
plt.legend()
plt.show()
