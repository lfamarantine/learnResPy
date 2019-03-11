# Importing Data into Python (Part 1)
# -----------------------------------
import pickle
import numpy as np
import pandas as pd
import os # for exploring working directory or operating system interfaces
from sas7bdat import SAS7BDAT

# 1. Introduction and flat files..
# --------------------------------

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

# 2. Importing data from other file types..
# -----------------------------------------
# excel, matlab, sas, stata, hdf5
# native python file: pickled files (serialized; convert object to sequence of bytes or bytestream)

# save a dictionary into a pickle file..
favorite_color = {"lion": "yellow", "kitty": "red"}
pickle.dump(favorite_color, open("data/fav_col.p", "wb"))
# load pickle file..
favorite_color = pickle.load(open("data/fav_col.p", "rb")) # rb: read-only & binary
# or..
with open('data/fav_col.p', 'rb') as file:
    d = pickle.load(file)

# import excel spreadsheet..
data = pd.ExcelFile('data/battledeath.xlsx')
# print sheet names..
print(data.sheet_names)
# load a particular sheet as a dataframe (either by sheet-name or sheet-index)..
df1 = data.parse('2002')
df1 = data.parse(0)
print(df1.head())

# custom spreadsheet import..
# .. parse the 1st column of the 2nd sheet and rename the column..
xl = pd.ExcelFile('data/battledeath.xlsx')
df1 = xl.parse(0, skiprows=[0], names=['Country','AAM due to War (2002)'])

# get working directory
wd = os.getcwd()
# contents of directory..
os.listdir(wd)

# sas & stata files..















