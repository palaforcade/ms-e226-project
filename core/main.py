import pandas as pd
from constants.columns import COLUMNS_SUBSET


raw_data = pd.read_csv("../data/raw_dataset.csv")

data_subset = raw_data[COLUMNS_SUBSET]

print(data_subset.head())
