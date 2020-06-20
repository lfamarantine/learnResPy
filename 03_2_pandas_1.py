# Pandas Foundations
# ------------------
import numpy as np
import pandas as pd

# 1. Data ingestion & inspection..
# --------------------------------

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
help(pd.read_csv)
df = pd.read_csv('data/temperatures.csv')
df.info()
df.iloc[:,0:3] = df.iloc[:,0:3].apply(pd.to_numeric, errors='coerce')
df.plot(subplots=True)


# 2. Exploratory Data Analysis
# ----------------------------
import matplotlib as plt
from sklearn import datasets
iris = datasets.load_iris()


# data preparation..
df = pd.read_csv('data/file_clean.csv')
df2 = df.T
df2.head()
df2.columns = df2.iloc[0]
df2 = df2[1:]
df2['Month'] = df2.index
df2 = df2.reset_index(drop=True)
df2.reindex(columns=list(df2.columns.values))
df2.info()
df2.iloc[:,0:4] = df2.iloc[:,0:4].apply(pd.to_numeric, errors='coerce')

# line plot..
df2.plot(x='Month')
# scatter plot..
df2.plot.scatter(x='IBM', y='APPLE')
# box plot..
df2['IBM'].plot(kind='box')

# calculating summary stats..
print(df2['IBM'].max())
print(df2['IBM'].describe())

# construct the mean percentage per firm..
x0 = df2.mean(axis='rows')

# quantiles..
print(df2['IBM'].quantile([0.05, 0.95]))

# counts and filtering..
df2[df2['IBM']>=120].count()


# 3. Time series in Pandas
# ------------------------

# filtering with dates..
# ---
df = pd.read_csv('data/ebola.csv', parse_dates=True, index_col='Date')
df.head()

# .. specific date or range..
df.loc['2015-01-05']
df.loc['2015-01-02':'2015-01-05']
df.loc['2014-04-07':'2014-03-29']

# pandas datetime object..
dft = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M')

# reindexing files..
# ---
ts1 = pd.read_csv('data/ts1.csv', header=None, index_col=0)
ts2 = pd.read_csv('data/ts2.csv', header=None, index_col=0)
# ..reindexing with nan's where no data..
ts3 = ts2.reindex(ts1.index)
# ..reindexing with previous where no data..
ts4 = ts2.reindex(ts1.index, method="ffill")
# ..reindexing with complete set..
sum12 = ts1 + ts2

# resampling..
# ---
df = pd.read_csv('data/weather_data_austin_2010.csv', parse_dates=True, index_col='Date')
# .. downsample to 6h intervals and aggregate by mean..
df1 = df['Temperature'].resample('6h').mean()
# .. downsample to daily intervals and aggregate by count..
df2 = df['Temperature'].resample('D').count()

# filtering..
august = df.loc['2010-08-01':'2010-08-31'].Temperature

# rolling calculations..
# ---
unsmoothed = df['Temperature']['2010-08-01':'2010-08-15']
smoothed = unsmoothed.rolling(window=24).mean()
august = pd.DataFrame({'smoothed':smoothed, 'unsmoothed':unsmoothed})
august.plot()

# resampling and rolling calculations..
august = df['Temperature']['2010-08-01':'2010-08-31']
# .. resample to daily data, aggregating by max..
daily_highs = august.resample('D').max()
# .. rolling 7-day window with method chaining to smooth the daily high temperatures..
daily_highs_smoothed = daily_highs.rolling(window=7).mean()

# chaining and filtering..
# ---
df = pd.read_csv('data/austin_airport_departure_data_2015_july.csv', parse_dates=True, header=16)
dallas = df['DAL'].str.contains('DAL')


# create random dates..
def random_dates(start, end, n, unit='D', seed=None):
    if not seed:  # from piR's answer
        np.random.seed(0)

    ndays = (end - start).days + 1
    return start + pd.to_timedelta(
        np.random.randint(0, ndays, n), unit=unit
    )


x0 = {'date':random_dates(pd.to_datetime('2016-07-01'), pd.to_datetime('2016-07-12'), 8), 'ts1': list(range(1,9))}
ts1 = pd.DataFrame(x0)
ts1.index = ts1['date']
ts1 = ts1['ts1']
x1 = {'date':random_dates(pd.to_datetime('2016-07-01'), pd.to_datetime('2016-07-15'), 11), 'ts2': list(range(1,12))}
ts2 = pd.DataFrame(x1)
ts2.index = ts2['date']
ts2 = ts2['ts2']

# unsampling & interpolating..
ts1.resample('M').first()
ts1.resample('M').first().interpolate('linear')


# resampling data..
df = pd.read_csv('data/weather_data_austin_2010.csv', parse_dates=['Date'])
df.head()

# datetime methods..
df['Date'].dt.hour
# set timezone..
# df['Date'].dt.tz_localize('UK/Central')


population = pd.read_csv('data/world_population.csv', parse_dates=True, index_col='Year')


# 4. Case Study
# -------------




















