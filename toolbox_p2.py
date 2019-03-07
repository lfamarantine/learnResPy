# Python Data Science Toolbox (Part 2)
# ------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

# 1. Using iterators in PythonLand
# --------------------------------

# iterating over a string..
word = "de"
it = iter(word)
next(it)
next(it)
# iterating at once with *..
print(*it)
# iterating over a dictionary..
pythonistas = {'hugo': 'bowne-anderson', 'francis': 'castro'}
for key, value in pythonistas.items():
    print(key, value)


# enumerate..
mutants = ['charles xavier', 'bobby drake', 'kurt wagner', 'max eisenhardt','kitty pryde']
mutant_list = list(enumerate(mutants))
print(mutant_list)
# unpack and print the tuple pairs..
for index1, value1 in enumerate(mutants):
    print(index1, value1)
# change the start index..
for index2, value2 in enumerate(mutants, start=1):
    print(index2, value2)

# using zip..
aliases = ['prof x', 'iceman', 'nightcrawler', 'magneto', 'shadowcat']
powers = ['telepathy', 'thermokinesis', 'teleportation', 'magnetokinesis', 'intangibility']
mutant_data = list(zip(mutants, aliases, powers))
print(mutant_data)
mutant_zip = zip(mutants, aliases, powers)
print(mutant_zip)
# unpack the zip object and print the tuple values..
for value1, value2, value3 in mutant_zip:
    print(value1, value2, value3)
# unzip the objects..
z1 = zip(mutants, powers)
result1, result2 = zip(*z1)

# iterating over large files into memory..
result = []
for chunk in pd.read_csv('data/pop_data.csv', chunksize=1000):
    result.append(sum(chunk['pop']))
total = sum(result)

# processing large amount of data..
def count_entries(csv_file, c_size, colname):
    """Return a dictionary with counts of
    occurrences as value for each key."""

    # initialize an empty dictionary..
    counts_dict = {}
    # iterate over the file chunk by chunk..
    for chunk in pd.read_csv(csv_file, chunksize=c_size):
        # iterate over the column in DataFrame..
        for entry in chunk[colname]:
            if entry in counts_dict.keys():
                counts_dict[entry] += 1
            else:
                counts_dict[entry] = 1
    return counts_dict
result_counts = count_entries('data/tweets.csv', 10, 'lang')
print(result_counts)


# 2. List Comprehensions & Generators
# -----------------------------------
# ..list comprehensions collapse for loops for building lists into a single line
# a simple list comprehension..
nums = [12, 8, 21, 3, 16]
new_nums = [num + 1 for num in nums]
rng = [num + 3 for num in range(11)]

# nested loops..
pairs2 = [(num1, num2) for num1 in range(0, 2) for num2 in range(6, 8)]
len(pairs2)

# iterator: object that keeps state & produces the next value when you call next()
# iteratable: returns an iterator

# create a 5 x 5 matrix using a list of lists..
matrix = [[col for col in range(5)] for row in range(5)]
print(matrix)

# conditionals in comprehensions..
nums = [num ** 2 for num in range(10) if num % 2 == 0]
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']
new_fellowship = [member if len(member) >= 7 else "" for member in fellowship]

# dict comprehensions..
pos_neg = {num: -num for num in range(9)}
new_fellowship = {member:len(member) for member in fellowship}

# generating list comprehensions..
# -> not stored in memory, hence useful when working with very large sequences
nums = (num ** 2 for num in range(10) if num % 2 == 0)
type(nums)
for i in nums:
    print(i)

# generator function..
def num_sequence(n):
    """Generate values from 0 to n."""
    i = 0
    while i < n:
        yield i
        i += 1
result = num_sequence(5)
for i in result:
    print(i)

# exercise..
df = pd.read_csv('data/tweets.csv', usecols=['created_at'])
tweet_time = df['created_at']
# extract the clock time..
tweet_clock_time = [entry[11:19] for entry in tweet_time]


# 3. Case Study
# -------------
wdi_dat = pd.read_csv('data/world_ind_pop_data.csv')
feature_names = ['CountryName', 'CountryCode', 'IndicatorName', 'IndicatorCode', 'Year', 'Value']
row_vals = ['Arab World', 'ARB', 'Adolescent fertility rate (births per 1,000 women ages 15-19)', 'SP.ADO.TFRT', '1960', '133.56090740552298']
zipped_lists = zip(feature_names, row_vals)
rs_dict = dict(zipped_lists)

# write a function to do this..
def lists2dict(list1, list2):
    """Return a dictionary where list1 provides
    the keys and list2 provides the values."""
    zipped_lists = zip(list1, list2)
    rs_dict = dict(zipped_lists)
    return rs_dict

rs_fxn = lists2dict(feature_names, row_vals)
# transforming into dataframe..
row_lists = zip(rs_fxn, rs_fxn, rs_fxn, rs_fxn, rs_fxn)
list_of_dicts = [lists2dict(feature_names, sublist) for sublist in row_lists]
# Turn list of dicts into a DataFrame: df
df = pd.DataFrame(list_of_dicts)

# using generators to read streaming data..
# .. process the first 1000 rows of a file line by line, to create a dictionary of the counts of how
#  many times each country appears in a column in the dataset.
# open a connection to the file..
with open('data/world_ind_pop_data.csv') as file:
    # skip the column names..
    file.readline()
    # initialize an empty dictionary,,
    counts_dict = {}
    # process only the first 1000 rows..
    for j in range(1000):
        # split the current line into a list..
        line = file.readline().split(',')
        # get the value for the first column..
        first_col = line[0]
        # if the column value is in the dict, increment its value
        if first_col in counts_dict.keys():
            counts_dict[first_col] += 1
        # else, add to the dict and set value to 1
        else:
            counts_dict[first_col] = 1
# print the resulting dictionary
print(counts_dict)

# write a generator to read chunks..
def read_large_file(file_object):
    """A generator function to read a large file lazily."""
    while True:
        data = file_object.readline()
        if not data:
            break
        yield data

with open('data/world_ind_pop_data.csv') as file:
    gen_file = read_large_file(file)


# using pandas read_csv iterator for streaming data..
df_reader = pd.read_csv('data/world_ind_pop_data.csv', chunksize=10)
print(next(df_reader))
print(next(df_reader))


# example with filters & conditions..
urb_pop_reader = pd.read_csv('data/world_ind_pop_data.csv', chunksize=1000)
# get the first DataFrame chunk..
df_urb_pop = next(urb_pop_reader)
print(df_urb_pop.head())
# check out specific country..
df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == 'CEB']
# zip DataFrame columns of interest..
pops = zip(df_pop_ceb['Total Population'], df_pop_ceb['Urban population (% of total)'])
# turn zip object into list..
pops_list = list(pops)
print(pops_list)

# now iterating over the entire dataset & creating additional columns..
# initialize empty DataFrame..
data = pd.DataFrame()
# iterate over each DataFrame chunk..
for df_urb_pop in urb_pop_reader:
    # check out specific country..
    df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == 'CEB']
    # zip DataFrame columns of interest..
    pops = zip(df_pop_ceb['Total Population'],
               df_pop_ceb['Urban population (% of total)'])
    pops_list = list(pops)
    # use list comprehension to create new DataFrame column 'Total Urban Population'..
    df_pop_ceb['Total Urban Population'] = [int(tup[0] * tup[1] * 0.01) for tup in pops_list]
    # append DataFrame chunk to data..
    data = data.append(df_pop_ceb)

# plot urban population data..
data.plot(kind='scatter', x='Year', y='Total Urban Population')










