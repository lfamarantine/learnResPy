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

# full illustration of indexing..
sales = pd.read_csv('data/pd2_sales.csv', index_col='month')
sales.head()
# modified index..
new_idx = [k.upper() for k in sales.index]
sales.index = new_idx
sales.head()
# change index name & columns name..
sales.index.name = 'MONTHS'
sales.columns.name = 'PRODUCTS'
sales.head()

# create index from scratch
# ..removing index..
sales = sales.reset_index()
ind0 = sales.iloc[:,0]
del sales['MONTHS']
sales.index = ind0
sales.head()

# hierarchical indexing..
sales = pd.read_csv('data/pd2_sales.csv')
del sales['month']
sales['state'] = ['CA','CA','NY','NY','TX','TX']
sales['month'] = [1, 2, 1, 2, 1, 2]
sales.head()
sales = sales.set_index(['state', 'month'])
sales.head()

# extracting data with multi-index..
print(sales.loc[['CA', 'TX']])
print(sales['CA':'TX'])
print(sales.loc[('NY', 1)])
print(sales.loc[(['CA','TX'],2),:])
print(sales.loc[(slice(None), 2), :])


# 3. Rearranging and reshaping data
# ---------------------------------

# pivoting
# ---
users = pd.read_csv('data/pd2_users.csv')
users.head()

# pivoting a single variable..
users_piv = users.pivot(index='weekday', columns='city', values='visitors')
users_piv.head()
# pivot all variables..
users_piv = users.pivot(index='weekday', columns='city')
users_piv.head()

# stacking & unstacking DF..
# ---
users = pd.read_csv('data/pd2_users.csv', index_col=['city', 'weekday'])
users = users.iloc[:,1:3]
users.head()
# unstack..
byweekday = users.unstack(level='weekday')
print(byweekday)
bycity = users.unstack(level='city')
print(bycity)
# stack..
print(byweekday.stack(level='weekday'))
print(bycity.stack(level='city'))

# restoring index order..
newusers = bycity.stack(level='city')
# ..swap index levels..
newusers = newusers.swaplevel(0, 1)
print(newusers)
# ..sort index..
newusers = newusers.sort_index()
print(newusers)
# verify..
print(newusers.equals(users))

# melting DF..
# ---

# melt & add names for readability..
users = pd.read_csv('data/pd2_users.csv', index_col=['city','weekday'])
users = users.reset_index('weekday')
print(users.index.names)
print(users)
visitors = pd.melt(users, id_vars=['weekday'], value_name='visitors')
print(visitors)
# melt multiple columns..
skinny = pd.melt(users, id_vars=['weekday','city'], value_name='value')
# get key-value paris..
kv_pairs = pd.melt(users, col_level=0)
print(kv_pairs)

# pivoting tables
# .. pivot table allows you to see all of your variables as a function of 2 other variables
by_city_day = pd.pivot_table(users, index='weekday', columns='city')
count_by_weekday1 = users.pivot_table(index='weekday', aggfunc='count')
signups_and_visitors_total = users.pivot_table(index='weekday', aggfunc='sum', margins=True)


# 4. Grouping data
# ----------------
sales = pd.DataFrame({'weekday': ['Sun','Sun','Mon','Mon'],
                      'city': ['Austin','Dallas','Austin','Dallas'],
                      'bread': [139,237,326,456],
                      'butter': [20,45,70,98]})

# grouping & aggregation..
# ---
# boolean filter & count..
sales.loc[sales['weekday'] == 'Sun'].count()
# groupby & count..
sales.groupby('weekday').count()
# groupby & sum..
sales.groupby('weekday')['bread'].sum()
sales.groupby('weekday')[['bread','butter']].sum()
# adding new series to groupby..
customers = pd.Series(['Dave','Alice','Bob','Alice'])
sales.groupby(customers)['bread'].sum()
# categorical data..
# advantages: uses less memory, speeds up operations like groupby()
sales['weekday'].unique()
sales['weekday'] = sales['weekday'].astype('category')
sales['weekday']

# aggregations..
sales.groupby('city')[['bread','butter']].max()
# multiple aggregations..
sales.groupby('city')[['bread','butter']].agg(['max','sum'])
# custom aggregation..
def data_range(series):
    return  series.max() - series.min()
sales.groupby('city')[['bread','butter']].agg(data_range)

# custom aggregation with dictionaries..
sales.groupby('city')[['bread','butter']].agg({'bread':'sum', 'butter':data_range})


# grouping & transformation..
# ---
auto = pd.read_csv('data/auto-mpg.csv')
auto.head()


def zscore(series):
    return (series - series.mean()) / series.std()


# transform..
zscore(auto['mpg']).head()
# groupby year & transform..
auto.groupby('yr')['mpg'].transform(zscore).head()

# using apply for more complicated aggregations..
def zscore_with_year_and_name(group):
    df = pd.DataFrame(
        {'mpg': zscore(group['mpg']),
         'year': group['yr'],
         'name': group['name']
        }
    )
    return df


auto.groupby('yr').apply(zscore_with_year_and_name).head()

from scipy.stats import zscore

# groupby & filtering..
# ---
auto.groupby('yr')['mpg'].mean()
splitting = auto.groupby('yr')
type(splitting)
print(splitting.groups.keys())

# groupby object iteration..
for group_name, group in splitting:
    avg = group['mpg'].mean()
    print(group_name, avg)

# groupby object iteration & filtering..
for group_name, group in splitting:
    avg = group.loc[group['name'].str.contains('chevrolet'), 'mpg'].mean()
    print(group_name, avg)

# as a list comprehension object..
chevy_means = {year:group.loc[group['name'].str.contains('chevrolet'), 'mpg'].mean() for year, group in splitting}
pd.Series(chevy_means)

# boolean groupby..
chevy = auto['name'].str.contains('chevrolet')
auto.groupby(['yr', chevy])['mpg'].mean()


# 5. Bringing it all together
# ---------------------------

medals = pd.read_csv('data/pd2_all_medalists.csv')
# grouping, filtering & aggregating..
USA_edition_grouped = medals.loc[medals.NOC == 'USA'].groupby('Edition')
USA_edition_grouped['Medal'].count()

# using value_counts for ranking..
country_names = medals['NOC']
medal_counts = country_names.value_counts()
print(medal_counts.head(15))

# summarising & adding new columns..
counted = medals.pivot_table(index='NOC', columns='Medal', values='Athlete', aggfunc='count')
counted['totals'] = counted.sum(axis='columns')
counted = counted.sort_values('totals', ascending=False)
print(counted.head(15))

# unique combinations..
ev_gen = medals[['Event_gender', 'Gender']]
ev_gen_uniques = ev_gen.drop_duplicates()
print(ev_gen_uniques)

# exploring data using count..
medals_by_gender = medals.groupby(['Event_gender', 'Gender'])
medal_count_by_gender = medals_by_gender.count()
print(medal_count_by_gender)

# locating suspicious data..
sus = (medals.Event_gender == 'W') & (medals.Gender == 'Men')
suspect = medals[sus]
print(suspect)

# constructing alternative country rankings..
# ---

# 2 new DF methods:
# ---
# - idxmax(): row or column label where maximum value is located
# - idxmin(): row or column label where minimum value is located
auto['mpg'].idxmax()
auto[['mpg','cyl','displ']].idxmax(axis='columns')


# nunique() ro rank by distinct sports..
country_grouped = medals.groupby('NOC')
Nsports = country_grouped['Sport'].nunique()
Nsports = Nsports.sort_values(ascending=False)
print(Nsports.head(15))

# US vs USSR cold war comparison..
# medals overall..
during_cold_war = (medals['Edition'] >=1952 ) & (medals['Edition'] <=1988)
is_usa_urs = medals.NOC.isin(['USA', 'URS'])
cold_war_medals = medals.loc[during_cold_war & is_usa_urs]
country_grouped = cold_war_medals.groupby('NOC')
Nsports = country_grouped['Sport'].nunique()
print(Nsports)

# medals consistency..
medals_won_by_country = medals.pivot_table(index='Edition', columns='NOC', values='Athlete', aggfunc='count')
cold_war_usa_urs_medals = medals_won_by_country.loc[1952:1988, ['USA','URS']]
most_medals = cold_war_usa_urs_medals.idxmax(axis='columns')
print(most_medals.value_counts())


# reshaping data for visualization..
# ---
usa = medals[medals['NOC']=='USA']
usa_medals_by_year = usa.groupby(['Edition', 'Medal'])['Athlete'].agg('count')
usa_medals_by_year = usa_medals_by_year.unstack(level='Medal')
# ts plot..
usa_medals_by_year.plot()
# area plot..
usa_medals_by_year.plot.area()

# transforming medal to categorical variable..
medals.Medal = pd.Categorical(values=medals.Medal, categories=['Bronze', 'Silver', 'Gold'], ordered=True)




