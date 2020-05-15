# EXTERNAL DATA DOWNLOADS IN PYTHON
# ---------------------------------
import pandas as pd
import pandas_datareader as pdr
from pandas_datareader import data
import numpy as np
import datetime

# --- yahoo
compt = {'Amazon':'AMZN', 'Apple':'AAPL', 'Exxon':'XOM', 'Chevron':'CVX'}
data_source = 'yahoo'
start_date = '2013-01-01'
end_date = '2020-01-31'
df = data.DataReader(list(compt.values()), data_source, start_date, end_date)


# --- fred
df = pdr.get_data_fred('GS10')


# --- world bank
from pandas_datareader import wb

# find available fields
matches = wb.search('gdp.*capita.*const')
# all fields
matches = wb.search('')
# all country codes
country_cd = pd.DataFrame(wb.country_codes, columns=['val'])
# download data
dat = wb.download(indicator='NY.GDP.PCAP.KD', country=['US', 'CA', 'MX'], start=2005, end=2008)




