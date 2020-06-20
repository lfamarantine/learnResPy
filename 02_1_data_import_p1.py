# Importing Data into Python (Part 1)
# -----------------------------------
import pickle
import h5py
import numpy as np
import pandas as pd
import os # for exploring working directory or operating system interfaces
from sas7bdat import SAS7BDAT
import scipy.io
from sqlalchemy import create_engine


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

# import from sas & stata files..
# ..sas
with SAS7BDAT('data/sales.sas7bdat') as file:
    df_sas = file.to_data_frame()
print(df_sas.head())
# ..stata
df = pd.read_stata('data/disarea.dta')

# importing HDF5 files..
# HDF5: hierarchical data format version 5
# - standard for storing large quantities of numerical data
# - datasets can be hundreds of gigabytes or terabytes & can scale to exabytes

filename = 'data/LIGO.hdf5'
data = h5py.File(filename, 'r')
print(type(data))
# structure of hdf5 file..
for key in data.keys():
    print(key)
for key in data['meta'].keys():
    print(key)
print(data['meta']['Description'].value)
print(data['meta'])
# extracting data from hdf5 file..
group = data['strain']
for key in group.keys():
    print(key)
strain = data['strain']['Strain'].value
# access the data..
strain[:10000]

# importing matlab files..
filename = 'data/ja_data2.mat'
mat = scipy.io.loadmat(filename)
print(type(mat))
for key in mat.keys():
    print(key)
# dimensions of array..
print(np.shape(mat['cfpCyt']))


# 3. Working with relational databases in Python
# ----------------------------------------------
# Todd's 12 Rules/Commandments (12 rules actually but 1st rule is 0-indexed)

# connect to engine..
engine = create_engine('sqlite:///data/Chinook.sqlite')

# 1. option..
# ---
# get table names of db..
table_names = engine.table_names()
# connect..
con = engine.connect()
# query..
rs = con.execute("select * from Album")
# save results & change column-names..
df = pd.DataFrame(rs.fetchall())
df.columns = rs.keys()
# close connection..
con.close()

# 2. option..
# open engine in context manager..
with engine.connect() as con:
    rs = con.execute("select * from Album")
    df = pd.DataFrame(rs.fetchall()) # df = pd.DataFrame(rs.many(size=3))
    df.columns = rs.keys()

# 3. option..
df1 = pd.read_sql_query("SELECT * FROM Album", engine)

# check whether the 2 tables are the same..
print(df.equals(df1))







