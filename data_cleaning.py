# Cleaning Data in Python
# -----------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# 1. Exploring your data..
# ------------------------

# common data problems:
# - inconsistent column names
# - missing data
# - outliers
# - duplicate rows
# - untidy
# - need to process columns
# - col;umn types can signal unexpected data values

df = pd.read_csv('data/gapminder.csv')
# inspect dataset..
df.head()
df.tail()
df.columns
df.shape
df.info()

# frequency counts..
df.cont.value_counts(dropna=False) # cont: column-name

# summary statistics..
df.describe()
df['population'].describe()

# histogram..
df.population.plot('hist')

# slice data & plot..
df[df.population > 1000000000]

# boxplots..
df.boxplot(column='population', by='cont')

# scatterplot..
df.plot(kind='scatter', x='gdp_cap', y='life_exp', rot=70)


# 2. Tidy Data..
# --------------
# - "Tidy Data" paper by Hadley Wickham
# - formalize the way we describe the shape of data
# - gives us a goal when formatting data
# - standard way to organize data values within a dataset

df = pd.read_csv('data/airquality.csv')
# melt the df..
# ---
df_melt = pd.melt(df, id_vars=['Month', 'Day'])
# .. with renaming..
df_melt = pd.melt(df, id_vars=['Month', 'Day'], var_name='measurement', value_name='reading')
df_melt.head()

# pivot the df..
# ---
# pivot (no duplicates) & pivot-tables (aggregate duplicates)
df_pivot = df_melt.pivot_table(index=['Month', 'Day'], columns='measurement', values='reading')
df_pivot.head()
# reset the index of the df after pivoting..
df_pivot = df_pivot.reset_index()

# pivoting duplicate values..
df_pivot = df_melt.pivot_table(index=['Month', 'Day'], columns='measurement', values='reading', aggfunc=np.mean)
df_pivot = df_pivot.reset_index()

# splitting columns..
# ---
df = pd.read_csv('data/tb.csv')
df_melt = pd.melt(df, id_vars=['country', 'year'])
# create new columns from existing..
df_melt['gender'] = df_melt.variable.str[0]
df_melt['age_group'] = df_melt.variable.str[1:]
df_melt.head()

# advanced splitting..
df = pd.read_csv('data/ebola.csv')
df_melt = pd.melt(df, id_vars=['Date', 'Day'], var_name='type_country', value_name='counts')
# create new column from existing..
df_melt['str_split'] = df_melt.type_country.str.split('_')
df_melt['type'] = df_melt.str_split.str.get(0)
df_melt['country'] = df_melt.str_split.str.get(1)
print(df_melt.head())


# 3. Combining data for analysis..
# --------------------------------

df = pd.read_csv('data/nyc_uber_2014.csv')
uber1 = df[0:98]
uber2 = df[99:197]
uber3 = df[198:]
# combine dataframes..
uber_c = pd.concat([uber1, uber2, uber3])









