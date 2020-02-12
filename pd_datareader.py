# EXTERNAL DATA DOWNLOADS IN PYTHON
# ---------------------------------
import pandas as pd
from pandas_datareader import data
import numpy as np
import datetime

# --- yahoo
compt = {'Amazon':'AMZN', 'Apple':'AAPL', 'Exxon':'XOM', 'Chevron':'CVX'}
data_source = 'yahoo'
start_date = '2013-01-01'
end_date = '2020-01-31'
df = data.DataReader(list(compt.values()), data_source, start_date, end_date)


