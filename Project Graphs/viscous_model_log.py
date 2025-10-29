"""
Viscous Model (Log-Transformed RFR)

This version applies logarithmic scaling to the Roof Fall Rate (RFR)
to smooth out large variations. The log transform highlights the inverse
trend more clearly, reducing the effect of extreme values.

This helps visualize the gradual decline of RFR with increasing CMRR
— a typical viscous behavior in geomechanical systems.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_data

# Load data
df = load_data()

# Log-transform RFR safely (log1p handles zeros)
df["RFR_log"] = np.log1p(df["RFR"])

# Plot
plt.figure(figsize=(8, 6))
sns.regplot(
    x="CMRR",
    y="RFR_log",
    data=df,
    scatter_kws={"s": 60, "color": "orange", "alpha": 0.8, "edgecolor": "black"},
    line_kws={"color": "darkred", "linewidth": 2},
)

plt.title("Viscous Model: CMRR vs log(RFR)", fontsize=14, fontweight="bold", pad=10)
plt.xlabel("Coal Mine Roof Rating (CMRR)", fontsize=12)
plt.ylabel("log(RFR)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

plt.savefig("Project Graphs/saved graphs/viscous_model_log.png", dpi=300, bbox_inches="tight")
plt.show()
