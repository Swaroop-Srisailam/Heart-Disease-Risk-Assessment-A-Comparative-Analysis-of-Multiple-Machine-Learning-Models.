import pandas as pd
from pandas_profiling import ProfileReport
heart_train_data=pd.read_csv("heart_cleveland.csv")
heart_train = heart_train_data.copy()

# Renaming some of the columns
heart_train=heart_train.rename(columns={'condition': 'target'})
print(heart_train.head())
pro=ProfileReport(heart_train,explorative=True)
pro