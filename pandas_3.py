# Merging Dataframes with Pandas
# ------------------------------
import pandas as pd

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













