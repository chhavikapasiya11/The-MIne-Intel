"""
Correlation Bar Graph

This graph shows how strongly each input variable affects the Roof Fall Rate (RFR):

It takes the correlation of every feature with RFR and displays it as bars.

Positive bars → an increase in the feature tends to increase RFR.
A positive correlation means the factor contributes to more roof falls.

Negative bars → an increase in the feature tends to decrease RFR (inverse relationship).
A negative correlation (e.g., high CMRR → low RFR) means better roof conditions reduce fall rate.

The length of the bar shows how strong that effect is.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import load_data

# Load data from a common file "data_loader.py"
df = load_data()

# computing correlations with RFR
corr_values = df.corr()["RFR"].drop("RFR").sort_values(ascending=False)

# Plot
plt.figure(figsize=(9,6))
sns.barplot(
    x=corr_values.index,
    y=corr_values.values,
    palette="coolwarm",
    edgecolor="dimgray",
    dodge=False, 
    legend=False
)

plt.title("Correlation of Each Input with Roof Fall Rate (RFR)", fontsize=15, fontweight="bold", pad=12)

plt.title("Correlation of Each Input with Roof Fall Rate (RFR)", fontsize=15, fontweight="bold", pad=12)
plt.ylabel("Correlation Coefficient", fontsize=12)
plt.xlabel("Input Parameters", fontsize=12)
plt.xticks(rotation=0, fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()

#plt.savefig("Project Graphs/saved graphs/correlation_bar_graph.png", dpi=300, bbox_inches="tight")

plt.show()
