# Merging Dataframes with Pandas
# ------------------------------
import pandas as pd
import numpy as np

# 1. Preparing data
# -----------------

# reading multiple files..
# tools for data import:
# - pd.read_excel()
# - pd.read_html()
# - pd_read_json()

# reading 2 files manually..
df1 = pd.read_csv('data/pd3_sales-jan-2015.csv')
df2 = pd.read_csv('data/pd3_sales-feb-2015.csv')

# looping through a list of files & reading them..
filenames = ['data/pd3_sales-jan-2015.csv', 'data/pd3_sales-feb-2015.csv']
dataframes = []
for f in filenames:
    dataframes.append(pd.read_csv(f))
# alternatively, use list comprehension..
dataframes = [pd.read_csv(f) for f in filenames]

# using glob module (wildcard functionality)..
# ---
from glob import glob
filenames = glob('data/pd3_sales*.csv')
dataframes = [pd.read_csv(f) for f in filenames]
dataframes[0].head()


# reindexing DF's..
# ---
weather1 = pd.read_csv('data/pd2_pittsburgh2013.csv', index_col='Date', parse_dates=True)
year = ['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
# reindex weather1 using the list year..
weather2 = weather1.reindex(year)
print(weather2)
# reindex weather1 using the list year with forward-fill..
weather3 = weather1.reindex(year).ffill()

# additional computations..
# ---
# ..1st example..
gdp = pd.read_csv('data/pd3_gdp_usa.csv', index_col='DATE', parse_dates=True)
# slice all the gdp data from 2008 onward..
post2008 = gdp['2008-01-01':]
print(post2008.tail(8))
# resample post2008 by year, keeping last..
yearly = post2008.resample('A').last()
print(yearly)
# compute percentage growth..
yearly['growth'] = yearly.pct_change() * 100
print(yearly)

# 2nd example..
sp500 = pd.read_csv('data/pd3_sp500.csv', index_col='Date', parse_dates=True)
exchange = pd.read_csv('data/pd3_exchange.csv', index_col='Date', parse_dates=True)
exchange.head()
dollars = sp500[['Open', 'Close']]
pounds = dollars.multiply(exchange['GBP/USD'], axis='rows')
print(pounds.head())


# arithmetic with series & DF's..
# ---
weather = pd.read_csv('data/pd2_pittsburgh2013.csv', index_col='Date', parse_dates=True)
weather.head()
weather.loc['2013-7-1':'2013-7-7', 'PrecipitationIn']
# scalar multiplication..
weather.loc['2013-7-1':'2013-7-7', 'PrecipitationIn'] * 2.54
week1_range = weather.loc['2013-7-1':'2013-7-7', ['Min TemperatureF','Max TemperatureF']]
week1_mean = weather.loc['2013-7-1':'2013-7-7', ['Mean TemperatureF']]
# division..
week1_range.divide(week1_mean, axis='rows')
# percentage change..
week1_mean_chng = week1_mean.pct_change() * 100


# 2. Concatenating data
# ---------------------

# .append(): series & DF method
# pd.concat(): pandas module function & can append row- & column-wise -> accepts a series eg. pd.concat([s1,s2,s3])
# concat is equivalent to using a series of .append()-calls

medals = pd.read_csv('data/pd3_medals.csv', index_col=0)

medal_types = ['bronze', 'silver', 'gold']
medals = []
# medal = 'bronze'
for medal in medal_types:
    file_name = "data/pd3_%s_top5.csv" % medal
    medal_df = pd.read_csv(file_name, index_col='Country')
    medals.append(medal_df)

# concatenate medals..
medals = pd.concat(medals, keys=['bronze', 'silver', 'gold'])

# slicing multi-indexed DF's..
medals_sorted = medals.sort_index(level=0)
# slicing..
print(medals_sorted.loc[('bronze','Germany')])
print(medals_sorted.loc['silver'])
# create alias for pd.IndexSlice
idx = pd.IndexSlice
# all the data on medals won by the United Kingdom..
print(medals_sorted.loc[idx[:,'United Kingdom'], :])


# outer & inner joins..
# ---
A = np.arange(8).reshape(2,4)+0.1
print(A)
B = np.arange(6).reshape(2,3)+0.2
print(B)
C = np.arange(12).reshape(3,4)+0.3
print(C)

# stack arrays horizontally..
np.hstack([B, A])
# ..or..
np.concatenate([B, A], axis=1)
# stack arrays vertically..
np.vstack([A, C])
# ..or..
np.concatenate([A, C], axis=0)

# joins..
bronze = pd.read_csv('data/pd3_Bronze.csv', index_col=0)
silver = pd.read_csv('data/pd3_Silver.csv', index_col=0)
gold = pd.read_csv('data/pd3_Gold.csv', index_col=0)
medal_list = [bronze, silver, gold]
# concatenate medal_list horizontally using an inner join..
medals = pd.concat(medal_list, axis=1, keys=['bronze', 'silver', 'gold'], join='inner')
print(medals)


# 3. Merging data
# ---------------
bronze = pd.read_csv('data/pd3_Bronze.csv', index_col=0)
silver = pd.read_csv('data/pd3_Silver.csv', index_col=0)
gold = pd.read_csv('data/pd3_Gold.csv', index_col=0)

# inner join..
pd.merge(bronze, gold)
# on multiple keys using suffixes..
pd.merge(bronze, gold, on=['NOC','Country'], suffixes=['_bronze', '_gold'])
# merging when column names don't match..
pd.merge(bronze, gold, left_on=['NOC','Country'], right_on=['NOC','Country'])
# alternative joins: left, right, inner, outer..
pd.merge(bronze, gold, how='right')

# ordered mergers..
software = pd.read_csv('data/pd3_feb-sales-Software.csv', parse_dates=['Date']).sort_values('Date')
hardware = pd.read_csv('data/pd3_feb-sales-Hardware.csv', parse_dates=['Date']).sort_values('Date')
# join yields an empty DF since it attempts to join on all columns..
pd.merge(software, hardware)
# use outer..
pd.merge(software, hardware, how='outer').sort_values(['Date'])
# .. alternatively use merge.ordered():
pd.merge_ordered(hardware, software)
pd.merge_ordered(hardware, software, on=['Date','Company'], suffixes=['_hardware','_software']).head()

# also supports filling null values..
auto = pd.read_csv('data/pd3_automobiles.csv')
oil = pd.read_csv('data/pd3_oil_price.csv', parse_dates=['Date'])
stocks = pd.read_csv('data/pd3_sp500.csv', parse_dates=['Date'])
pd.merge_ordered(stocks, oil, on='Date', fill_method='ffill')

# merge values in order using the on column, but for each row in the left DF, only rows from the right DF whose 'on'
# column values are less than the left value will be kept..
merged = pd.merge_asof(auto, oil, left_on='yr', right_on='Date')


# Pandas Accessors
# ----------------









