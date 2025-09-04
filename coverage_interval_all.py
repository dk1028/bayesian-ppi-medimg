import matplotlib.pyplot as plt
import pandas as pd

# Redefine the data (after session reset)
data2 = {
    'prior': ['uniform'] * 4 + ['jeffreys'] * 4,
    'n_labels': [10, 20, 40, 80] * 2,
    'chain_cov': [0.96, 1.00, 0.94, 0.94, 1.00, 1.00, 0.90, 0.92],
    'chain_w': [0.396875, 0.271335, 0.189356, 0.131579, 0.369410, 0.262156, 0.178783, 0.122471],
    'naive_cov': [0.92, 0.96, 0.96, 0.94, 0.98, 0.90, 0.92, 0.94],
    'naive_w': [0.481963, 0.362300, 0.272014, 0.196974, 0.484546, 0.357574, 0.269901, 0.199326],
    'diff_cov': [0.56, 0.68, 0.88, 0.88, 0.48, 0.72, 0.86, 0.90],
    'diff_w': [0.288050, 0.202000, 0.174525, 0.129263, 0.238000, 0.208075, 0.164525, 0.123537]
}

df2 = pd.DataFrame(data2)

# Visualization: Coverage
plt.figure(figsize=(10, 5))
for prior in ['uniform', 'jeffreys']:
    subset = df2[df2['prior'] == prior]
    plt.plot(subset['n_labels'], subset['chain_cov'], 'o-', label=f'Chain ({prior})')
    plt.plot(subset['n_labels'], subset['naive_cov'], 's--', label=f'Naive ({prior})')
    plt.plot(subset['n_labels'], subset['diff_cov'], 'd-.', label=f'Diff ({prior})')
plt.ylim(0.4, 1.05)
plt.xlabel("Number of Labels")
plt.ylabel("Coverage")
plt.title("Coverage vs. Number of Labels")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualization: Interval Width
plt.figure(figsize=(10, 5))
for prior in ['uniform', 'jeffreys']:
    subset = df2[df2['prior'] == prior]
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
