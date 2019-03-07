# Importing Data into Python (Part 1)
# -----------------------------------

import pandas as pd
import numpy as np

# reading a text file..
# ---
filename = 'data/seaslug.txt'
file = open(filename, mode='r') # 'r' / 'w' is read / write
text = file.read()
file.close()

# context managers with..
with open('data/seaslug.txt','r') as file:
    print(file.read())
# ..

# import flat files..
# ---
# ..flat files: text files containing records (eg. table data)
# -> record: row of fields | column: feature
# types: .csv, .txt
# -> if only numbers: import via numpy else: pandas

# BDFL's guiding principles for Python's design into 20 aphorisms..
import this


# import flat files using numpy..
# - numpy is standard for storing numerical data
# - essential for other packages: scikit-learn

# read specific data (2nd & 3rd column + skip 1st row)..
data = np.loadtxt('data/MNIST.csv', delimiter=',', skiprows=1, usecols=[0, 2])
# import as float and skip 1st row..
data_float = np.loadtxt('data/seaslug.txt', delimiter='\t', dtype=float, skiprows=1)

# loadtxt() will throw errors when working with different datatypes, use np.genfromtxt()..
data = np.genfromtxt('data/titanic_sub.csv', delimiter=',', names=True, dtype=None)

# alternative importing functions from numpy..
d = np.recfromcsv('data/MNIST.csv')
print(d[:3])

# import using pandas (1st 5 rows, no headers)..
data = pd.read_csv('data/gapminder.csv', nrows=5)
data.head()

# transform to numpy array..
data_array = data['life_exp'].values

# customizing pandas import to allow for NA' handling, empty lines or comments..
data = pd.read_csv('data/titanic_sub.csv', sep=',', comment='#', na_values='NaN')





