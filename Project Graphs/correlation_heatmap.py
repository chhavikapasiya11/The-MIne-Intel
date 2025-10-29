"""
# This heatmap visualizes the linear relationships among all major input features 
# (CMRR, PRSUP, Depth, IS, MH) and the output Roof Fall Rate (RFR).
# The color scale indicates the strength and direction of correlation:
# red = negative correlation (inverse relation), blue = positive correlation.
# It helps identify which parameters most influence roof fall behavior.

"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import load_data

# Load data from a common file "data_loader.py"
df = load_data()

# Correlation heatmap
corr_matrix = df.corr()

plt.figure(figsize=(8, 6))  
sns.heatmap(
    corr_matrix,
    annot=True,             # Show numeric correlation values
    cmap='coolwarm',        # Color theme (red = negative, blue = positive)
    fmt=".2f",              # 2 decimal precision
    linewidths=0.6,         # Light grid lines between cells
    square=True,            # Keep cells square-shaped
    cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}
)

plt.title("Correlation Heatmap: Major Inputs and Roof Fall Rate", fontsize=15, fontweight="bold", pad=15)
plt.xticks(rotation=0, fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()

plt.savefig("Project Graphs/saved graphs/correlation_heatmap.png", dpi=300, bbox_inches="tight")  # save inside folder
plt.show() 