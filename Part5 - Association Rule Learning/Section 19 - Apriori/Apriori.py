import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Data preprocessing
data_set = pd.read_csv(r"C:\Users\youne\Desktop\AI_certif\Part5 - Association Rule Learning\Data\Market_Basket_Optimisation.csv", header = None)
transactions = data_set.astype(str).values.tolist()
print(transactions)

#Training the Apriori model
from apyori import apriori
associations = apriori(
    transactions = transactions,      # your list of transactions
    min_support = 21/7501,            # itemset must appear at least 21 times
    min_confidence = 0.2,             # rule must be correct at least 20% of the time
    min_lift = 3,                     # rule must be at least 3x stronger than random chance
    min_length = 2,                   # minimum number of items in rule or Association
    max_length = 2                    # maximum number of items in rule (exactly 2 here)
)
# Convert generator object into a list so we can iterate and inspect it
results = list(associations)

# Function to extract clean readable values from the messy Apriori output
def inspect(results):
    rows = []  # will store cleaned rules
    # Each result represents one frequent itemset
    for result in results:
        
        # Each itemset can generate multiple rules (A→B, B→A)
        for stat in result.ordered_statistics:
            
            rows.append((
                tuple(stat.items_base)[0],  # Left-hand side (A)
                tuple(stat.items_add)[0],   # Right-hand side (B)
                result.support,             # Support of the full itemset
                stat.confidence,            # Confidence of the rule
                stat.lift                   # Lift of the rule
            ))
    
    return rows


# Convert cleaned rules into a pandas DataFrame (table format)
resultsinDataFrame = pd.DataFrame(
    inspect(results),
    columns = [
        'Left Hand Side',
        'Right Hand Side',
        'Support',
        'Confidence',
        'Lift'
    ]
)

# Show the 10 strongest rules sorted by highest Lift
resultsinDataFrame.nlargest(
    n = 10,              # number of rows to display
    columns = 'Lift'     # sort based on Lift column
)
