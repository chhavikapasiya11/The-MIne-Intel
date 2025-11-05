import pandas as pd

# one reusable function to get the same cleaned dataset anytime.
def load_data(filepath="original_data.xlsx"):
    #Load and return the cleaned Mine-Intel dataset
    data = pd.read_excel(filepath)
    
    selected_columns = [
        "CMRR", "PRSUP", 
        "depth_of_ cover",
        "intersection_diagonal", 
        "mining_hight", 
        "roof_fall_rate"
    ]

    df = data[selected_columns]
    df = df.rename(columns={
        "depth_of_ cover": "Depth",
        "mining_hight": "MH",
        "intersection_diagonal": "IS",
        "roof_fall_rate": "RFR",
        "PRSUP": "PRSUP"
    })

    return df