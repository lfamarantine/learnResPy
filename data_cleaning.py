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

# concatenating..
# ---
df = pd.read_csv('data/nyc_uber_2014.csv')
uber1 = df[0:98]
uber2 = df[99:197]
uber3 = df[198:]
# combine dataframes..
# bind rows..
uber_c = pd.concat([uber1, uber2, uber3])
# bind cols..
uber1 = df.iloc[:,[1]]
uber2 = df.iloc[:,[2]]
uber_c = pd.concat([uber1, uber2], axis=1)

# globbing in python..
# ---
# wildcards: *? -> eg. any csv file: *.csv | any single character: file_?.csv
import glob

# import all csv files into a list..
csv_files = glob.glob('data/*.csv')
print(csv_files)

# load & append all csv files in a list..
list_data = []
for filename in csv_files:
    data = pd.read_csv(filename)
    list_data.append(data)

# merging data
# ---
# 3 different types of mergers; One-to-one | Many-to-one / one-to-many | Many-to-many
# one-to-one merge..
tmp = {'state': ['California', 'Texas', 'Florida', 'New York'] , 'population_2016': [392, 278, 206, 197]}
tmp2 = {'name': ['California', 'Texas', 'Florida', 'New York'], 'ANSI': ['CA', 'FL', 'NY', 'TX']}
A = pd.DataFrame(tmp)
B = pd.DataFrame(tmp2)
pd.merge(left=A, right=B, on=None, left_on='state', right_on='name')


# 4. Cleaning data for analysis..
# -------------------------------

# converting data types..
# ---
df = pd.read_csv('data/tips.csv')
df.info()
df.head()
df['time'] = df['time'].astype('category')
df.info()
# converting categorical data to 'category' dtype:
# - can make the dataframe smaller in memory
# - can make them be utilized by other python libraries for analysis

# numerical data loaded as a string..
# ---
df = pd.read_csv('data/tips_bad.csv')
df.info()
df['tip'] = pd.to_numeric(df['tip'], errors='coerce')
# or..
df.time = df.time.astype(str)
df.info()

# regular expressions..
# ---
import re
# example matches..
# 17        \d*
# $17       \$\d*
# $17.00    \$\d*\.\d*
# $17.89    \$\d*\.\d{2}
# $17.895   ^\$\d*\.\d{2}$
pattern = re.compile('\$\d*\.\d{2}')
result = pattern.match('$17.89')
bool(result)

# 2nd example: compile a pattern that matches a phone number of format xxx-xxx-xxxx
# .. using \d{x} to match x digits
prog = re.compile('\d{3}-\d{3}-\d{4}')
# see if the pattern matches..
result = prog.match('123-456-7890')
print(bool(result))
# zee if the pattern matches..
result2 = prog.match('1123-456-7890')
print(bool(result2))

# extracting numerical values from strings..
# .. \d to find digits & + so that the previous element is matched one or more times..
matches = re.findall('\d+', 'the recipe calls for 10 strawberries and 1 banana')
print(matches)

# find pattern with capital letter as the 1st letter followed by arbitrary number of characters..
pattern3 = bool(re.match(pattern='[A-Z]\w*', string='Australia'))
print(pattern3)

# use functions to clean data..
# ---
tmp = {'treatment a': [18, 12, 24], 'treatment b': [42, 31, 27]}
df = pd.DataFrame(tmp)
# apply function..
# column-wise..
df.apply(np.mean, axis=0)
# row-wise..
df.apply(np.mean, axis=1)

# custom functions to clean data..
df = pd.read_csv('data/tips.csv')
df.head()

# a custom function to create binary variable from a string variable..


def recode_gender(gender):
    # return 0 if gender is 'Female'
    if gender == 'Male':
        return 1
    # return 1 if gender is 'Male'
    elif gender == 'Female':
        return 0
    # return np.nan
    else:
        return np.nan


# add recode variable derived from existing columns..
df['recode'] = df.sex.apply(recode_gender)

# using lambda functions to clean data..
# ---







