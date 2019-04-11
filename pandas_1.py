# Pandas Foundations
# ------------------

import pandas as pd
import numpy as np

# 1. Data ingestion & inspection..
# --------------------------------
import pandas as pd

# building data from scratch..
list_keys = ['Country', 'Total']
list_values = [['United States', 'Soviet Union', 'United Kingdom'], [1118, 473, 273]]
# list..
zipped = list(zip(list_keys,list_values))
# dictionary..
data = dict(zipped)
# dataframe..
df = pd.DataFrame(data)

# set column names..
df.columns = ['US', 'RU', 'UK']

df_tmp = pd.read_csv('data/gapminder.csv')
df = df_tmp.iloc[:, [3]]
df.index = df_tmp['country']

# create array of DataFrame values..
np_vals = df.values
# create new array of base 10 logarithm values..
np_vals_log10 = np.log10(np_vals)

# create array of new DataFrame by passing df to np.log10()..
df_log10 = np.log10(df)

# cleaning, naming & parsing data..
# pd.read_csv(filepath, header=None, names=col_names, na_values={'sunspots': [' -1']}, parse_dates=[[0, 1, 2]])
# exporting data..
df.to_csv(df)

# reading a flat file..
data_file = 'data/world_population.csv'
new_labels = ['year','population']
df2 = pd.read_csv(data_file, header=0, names=new_labels)

# delimiters, headers & extensions..
# ---
file_messy = 'data/messy_stock_data.tsv.txt'
df1 = pd.read_csv(file_messy)
print(df1.head())
df2 = pd.read_csv(file_messy, delimiter=' ', header=3, comment='#')
print(df2.head())

# save as csv..
df2.to_csv(r'data/file_clean.csv', index=False)
# save as excel..
df2.to_excel('data/file_clean.xlsx', index=False, sheet_name='s0')

# plotting with pandas..
# ---
import matplotlib.pyplot as plt
df2.plot(color='red', title='Temperature in Austin')
plt.xlabel('Hours since midnight August 1, 2010')
plt.ylabel('Temperature (degrees F)')


# 2. Exploratory Data Analysis
# ----------------------------









