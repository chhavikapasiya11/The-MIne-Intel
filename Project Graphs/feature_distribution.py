"""
Feature Distribution Plot

This figure shows how each key feature (CMRR, PRSUP, Depth of Cover, Intersection Diagonal, Mining Height, and Roof Fall Rate) is distributed in the dataset.

The histograms display how frequently different values occur, while the red dashed line marks the mean of each feature.

These plots help in understanding data spread, identifying skewed or uneven distributions, and spotting any unusual or extreme values before applying machine learning models.

"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import load_data

# Load data from a common file "data_loader.py"
df = load_data()

# makes sure graphs display fully
pd.set_option('display.max_columns', None)

# Plot feature distributions with mean lines
plt.figure(figsize=(12, 8))

for i, col in enumerate(df.columns, 1):
    plt.subplot(2, 3, i)
    plt.hist(df[col], bins=10, color="skyblue", edgecolor="black")

    plt.axvline(df[col].mean(), color='red', linestyle='--', linewidth=1)       # shows mean lines

    plt.title(col, fontsize=11)
    plt.xlabel(col, fontsize=9)
    plt.ylabel("Count", fontsize=9)
    plt.grid(alpha=0.3)

plt.suptitle("Feature Distributions", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])

#plt.savefig("Project Graphs/saved graphs/feature_distributions.png", dpi=300, bbox_inches="tight")
plt.show()