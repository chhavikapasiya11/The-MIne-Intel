"""
Viscous Model: Relationship Between CMRR and Roof Fall Rate (RFR)

This graph represents the inverse relationship between rock mass strength (CMRR) 
and the roof fall rate (RFR). As CMRR increases, the roof becomes more stable,
resulting in a decrease in roof fall rate.

This "viscous model" analogy comes from rock mechanics, where weaker roof strata
behave like a viscous material, showing higher deformation or fall rate.

"""

import seaborn as sns
import matplotlib.pyplot as plt
from data_loader import load_data

# --- Load dataset ---
df = load_data()

# --- Create scatter + regression plot ---
plt.figure(figsize=(8, 6))
sns.regplot(
    x="CMRR",
    y="RFR",
    data=df,
    scatter_kws={"s": 60, "color": "steelblue", "alpha": 0.8, "edgecolor": "black"},
    line_kws={"color": "red", "linewidth": 2},
)

# --- Customize graph ---
plt.title("Viscous Model: Relationship Between CMRR and Roof Fall Rate (RFR)", fontsize=14, fontweight="bold", pad=12)
plt.xlabel("Coal Mine Roof Rating (CMRR)", fontsize=12)
plt.ylabel("Roof Fall Rate (RFR)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# --- Save & Show ---
plt.savefig("Project Graphs/saved graphs/viscous_model.png", dpi=300, bbox_inches="tight")
plt.show()


"""
Blue points → actual data (mine samples).

Red line → fitted regression (trendline showing inverse relation).

As CMRR increases → RFR generally decreases → confirms viscous model behavior.

Small irregularities = natural geological variations.

"""