# Pandas Foundations
# ------------------

# 1. Data ingestion & inspection..
# --------------------------------
import pandas as pd

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























