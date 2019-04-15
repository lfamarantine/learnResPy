# Manipulating dataframes with pandas
# -----------------------------------

import pandas as pd

# covered:
# - extracting, filtering & transforming DF in pandas
# - advanced indexing at multiple levels
# - tidying, restructuring & rearranging data
# - pivoting, melting & stacking DF
# - identifying & splitting DF by groups


# 1. Extracting and transforming data
# -----------------------------------
df = pd.read_csv('data/pd2_sales.csv', index_col='month')
df.head()

# indexing..
df['salt']['Jan']
df.eggs['Mar']
df.loc['May', 'spam']
df.iloc[4, 2]
# selecting only some columns as DF..
df[['salt', 'spam']]

# slicing DF..
type(df['eggs'])
df['eggs'][4]
df['eggs'][1:4]
df.loc[:,'eggs':'salt'] # all rows, some columns
df.loc['Jan':'Apr',:] # some rows, all columns
df['Jan':'May':-1] # slice in reverse order
df.iloc[2:5, 1:]
df.loc['Jan':'May', ['eggs', 'spam']] # using lists rather than slices
df.iloc[[0, 4, 5], 0:2]

# slicing series vs DF..
df['eggs'] # series
df[['eggs']] # DF


# filtering DF..
df[(df.salt > 60) | (df.salt < 20)]
df2 = df.copy()
df2['bacon'] = [0, 0, 50, 60, 70, 80]
df2.loc[:, df2.all()] # columns with all nonzeros
df2.loc[:, df2.any()] # columns with any nonzeros
df.loc[:, df.isnull().any()] # columns with any nan's
df.dropna(how='any') # remove rows with any na's

# modifying a column based on another..
df.eggs[df.salt > 55] += 5



















