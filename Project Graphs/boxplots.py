"""
Boxplots for Input Features

This visualization shows the spread and presence of outliers in each major input feature.
It helps verify data consistency and whether any mining parameter has unusually high or low values.

Each box represents one feature:
- Line in middle → median value
- Dots beyond whiskers → potential outliers
- Tall boxes → higher variation in that feature
"""

import seaborn as sns
import matplotlib.pyplot as plt
from data_loader import load_data

# --- Load cleaned dataset ---
df = load_data()

# --- Create boxplot ---
plt.figure(figsize=(10, 8))
sns.boxplot(
    data=df[["CMRR", "PRSUP", "Depth", "IS", "MH"]],
    palette="coolwarm",
    width=0.6,
    fliersize=5,
    linewidth=1.2
)

plt.title("Boxplot: Spread of Major Input Features", fontsize=15, fontweight="bold", pad=10)
plt.ylabel("Value Range", fontsize=12)
plt.xlabel("Input Features", fontsize=12)
plt.xticks(rotation=0, fontsize=11)
plt.yticks(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Make y-axis taller & add spacing
plt.ylim(0, df[["CMRR","PRSUP","Depth","IS","MH"]].max().max() * 1.1)

plt.tight_layout()
plt.savefig("Project Graphs/saved graphs/boxplot_features.png", dpi=300, bbox_inches="tight")
plt.show()


"""
“Boxplot analysis was used to study the spread and outliers across key input parameters.
Depth showed the highest variation, with a few extreme but valid observations corresponding to deeper mine sections.
Other parameters remained within stable ranges, indicating consistent data quality.”
"""

