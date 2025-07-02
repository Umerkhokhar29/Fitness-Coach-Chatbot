import pandas as pd

dfNew = pd.read_csv("fitness_data_fixed2.csv")

print(dfNew.head())

print(dfNew.info())
print(dfNew.describe())
