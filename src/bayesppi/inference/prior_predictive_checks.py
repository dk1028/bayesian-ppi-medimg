import numpy as np
import matplotlib.pyplot as plt

# Prior predictive simulation
rng = np.random.default_rng(42)
n = 20000
theta_A = rng.beta(1, 1, n)
theta_H1 = rng.beta(1, 1, n)
theta_H0 = rng.beta(1, 1, n)
g = theta_A * theta_H1 + (1 - theta_A) * theta_H0

# Plot histogram of g
plt.figure(figsize=(6,4))
plt.hist(g, bins=50, density=True)
plt.axvline(0.5, linestyle='--')   # mean of g under the prior
plt.xlabel(r"$g = \theta_A \theta_{H|1} + (1-\theta_A)\theta_{H|0}$")
plt.ylabel("Density")
plt.title("Prior predictive distribution of $g$")
plt.tight_layout()
plt.savefig('/mnt/data/prior_predictive_checks.png')
'/mnt/data/prior_predictive_checks.png'
