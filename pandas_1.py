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





