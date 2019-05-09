# Manipulating dataframes with pandas
# -----------------------------------

import pandas as pd
import numpy as np

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

# transforming data..
df.head()
# .. DF-vectorized methods..
df.floordiv(12) # divide by 12 and round to floor
# .. numpy-vectorized methods..
np.floor_divide(df, 12) # divide by 12 and round to floor

# plain python function..
def dozens(n):
    return n//12

df.apply(dozens) # convert to dozens unit

# lambda functions..
df.apply(lambda n: n//12)

# storing a transformation..
df['dozen_of_eggs'] = df.eggs.floordiv(12)

# working with string values..
df.index = df.index.str.upper()
df.head()
df.index = df.index.map(str.lower)
df.head()

# defining columns using other columns..
df['salty_eggs'] = df.salt + df.dozen_of_eggs

# advanced transformations..
# ---
election = pd.read_csv('data/pd2_pennsylvania2012_turnout.csv')
election.head()
# mapping with another vector..
red_vs_blue = {'Obama':'blue', 'Romney':'red'}
election['color'] = election.winner.map(red_vs_blue)
election.head()

# vectorized functions..
from scipy.stats import zscore
election['turnout_zscore'] = zscore(election['turnout'])
election.head()


# 2. Advanced indexing
# --------------------

# indexes:
# - immutable
# - homogenous in data type

prices = [10.7, 10.86, 10.74, 10.71, 10.79]
shares = pd.Series(prices)
print(shares)

days = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri']
shares = pd.Series(prices, index=days)
print(shares)

# slice from index..
print(shares.index[2])
print(shares.index[:2])

# modifying index name..
shares.index.name = 'weekday'
print(shares)

# modifying index entries..
shares.index[1] = 'Wednesday' # throws an error bcz not mutable
# ..only possible by modifying all indexes at once..
shares.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
print(shares)

# assigning the index..
df = pd.read_csv('data/tb.csv')
df.head()
df.index = df['country']
df.head()
del df['country']
df.head()

# get index right from the start..
df = pd.read_csv('data/tb.csv', index_col='country')
df.head()

# hierarchical indexing..
# ---
df = pd.read_csv('data/tb.csv')
df.head()
# .. 2 indexes..
df = df.set_index(['country','year'])
df.head()
# show indexes..
print(df.index.names)
# sorting indexes..
df = df.sort_index()
df.head()
# filtering with indexes..
df.loc['AF']


# 3. Rearranging & reshaping data
# -------------------------------

# melting..
df = pd.read_csv('data/pd2_users.csv', index_col=['weekday','city'])
df2 = pd.read_csv('data/pd2_users.csv')
df.head()
print(df.index.names)
# reset the index:..
visitors_by_city_weekday = df.reset_index('weekday')
# melt..
visitors = pd.melt(visitors_by_city_weekday, id_vars=['weekday'], value_name='visitors')
visitors.head()
# melt 2 variables..
skinny = pd.melt(df2, id_vars=['weekday','city'], value_name='value')
# key-value pairs..
kv_pairs = pd.melt(df, col_level=0)
print(kv_pairs)

# pivot tables..
by_city_day = pd.pivot_table(df, index='weekday', columns='city')
print(by_city_day)
# pivot table & count in each column..
count_by_weekday1 = df.pivot_table(index='weekday', aggfunc='count')
print(count_by_weekday1)
# pivot table & summarise by group & add total..
dft = df.pivot_table(index='weekday', aggfunc='sum', margins=True)
print(dft)


# 4. Grouping data
# ----------------
















