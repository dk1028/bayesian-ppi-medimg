import matplotlib.pyplot as plt
import pandas as pd

# Define data
data = {
    'prior': ['uniform'] * 4 + ['jeffreys'] * 4,
    'n_labels': [10, 20, 40, 80] * 2,
    'chain_cov': [1.00, 1.00, 0.98, 1.00, 1.00, 1.00, 0.98, 1.00],
    'chain_w': [0.378666, 0.258916, 0.193733, 0.144445, 0.344443, 0.253695, 0.182425, 0.139233],
    'naive_cov': [0.98, 1.00, 0.98, 1.00, 0.90, 0.98, 0.94, 0.98],
    'naive_w': [0.460678, 0.336439, 0.248861, 0.178753, 0.433806, 0.342746, 0.246771, 0.179330],
    'diff_cov': [0.58, 0.62, 0.94, 1.00, 0.44, 0.76, 0.96, 0.96],
    'diff_w': [0.202100, 0.139000, 0.144037, 0.110019, 0.160050, 0.172000, 0.137512, 0.105781]
}

df = pd.DataFrame(data)

# Visualization: Coverage
plt.figure(figsize=(10, 5))
for prior in ['uniform', 'jeffreys']:
    subset = df[df['prior'] == prior]
    plt.plot(subset['n_labels'], subset['chain_cov'], 'o-', label=f'Chain ({prior})')
    plt.plot(subset['n_labels'], subset['naive_cov'], 's--', label=f'Naive ({prior})')
    plt.plot(subset['n_labels'], subset['diff_cov'], 'd-.', label=f'Diff ({prior})')
plt.ylim(0.3, 1.05)
plt.xlabel("Number of Labels")
plt.ylabel("Coverage")
plt.title("Coverage vs. Number of Labels")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualization: Interval width
plt.figure(figsize=(10, 5))
for prior in ['uniform', 'jeffreys']:
    subset = df[df['prior'] == prior]
    plt.plot(subset['n_labels'], subset['chain_w'], 'o-', label=f'Chain ({prior})')
    plt.plot(subset['n_labels'], subset['naive_w'], 's--', label=f'Naive ({prior})')
    plt.plot(subset['n_labels'], subset['diff_w'], 'd-.', label=f'Diff ({prior})')
plt.xlabel("Number of Labels")
plt.ylabel("Interval Width")
plt.title("Interval Width vs. Number of Labels")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
