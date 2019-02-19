

# 1. Python Basics
# ----------------
# installing packages: python -m pip install pandas
# execute code via: shift+alt+e

import pandas as pd
import os
import numpy as np
import math

# current working directory..
print(os.getcwd())
# set working directory..
os.chdir('/Users/dariopopadic/PycharmProjects/datacamp/')

savings = 100
result = 100 * 1.10 ** 7

# type conversion..
print("I started with $" + str(savings) + " and now have $" + str(result) + ". Awesome!")


# 2. Python Lists
# ----------------

# create a list..
areas = [11.25, 18, 20, 10.75, 9.5]
# a list with different types..
areas = ['hallway', 11.25, 'kitchen', 18, 'living room', 20, 'bedroom', 10.75, 'bathroom', 9.5]

# list of lists..
house = [["hallway", 11.25],
         ["kitchen", 18],
         ["living room", 20],
         ['bedroom', 10.75],
         ['bathroom', 9.5]]

# subsetting lists.. [start:end] -> [inclusive:exclusive]
areas[0] == areas[-10]
downstairs = areas[:6]
upstairs = areas[6:]

# subsetting list of lists..
x = [["a", "b", "c"],
     ["d", "e", "f"],
     ["g", "h", "i"]]
x[2][0]

# manipulating lists..
areas[9] = 10.5
areas[4] = "chill zone"
print(areas)
areas_1 = areas + ["poolhouse", 24.5]
areas_2 = areas_1 + ["garage", 15.45]
# deleting list elements..
del(areas_2[10:11])

# inner workings of lists..
# creating a simple copy like areas_copy = areas & changing areas_copy will change areas too! Do a more
# explicit copy of the list with areas[:] or list()
areas = [11.25, 18.0, 20.0, 10.75, 9.50]
areas_copy = areas # original list changed
areas_copy = areas[:] # original list unchanged
areas_copy[0] = 5.0
print(areas)


# 3. Functions & Packages
# -----------------------
# function help-page..
help(round)

# optional arguments: [, ndigits] in round-function

# familiar functions..
var1 = [1, 2, 3, 4]
var2 = True
print(len(var1))
# convert var2 to an integer..
out2 = int(var2)

help(sorted)

# methods: call functions that belong to objects
# ---
# different object types have different methods associated with it..
# list methods: .index, .count, .append etc.
# floats, integers, booleans & strings also have methods associated: .capitalize, .replace etc.
# ..find what methods are available with: help(str)
room = "poolhouse"
room_up = room.upper()
print(room_up)
# Print out the number of o's in room..
print(room.count("o"))
# list methods..
areas = [11.25, 18.0, 20.0, 10.75, 9.50]
print(areas.index(20))
# print out how often 14.5 appears in areas..
print(areas.count(20))
areas.append(24.5)
areas.reverse()
print(areas)

# installing packages..
# 1. download pip-file: http://pip.readthedocs.org/en/stable/installing
# 2. terminal: get-pip.py

# working with packages..
# selective import: from math import pi
r = 0.43
C = 2 * math.pi * r
A = math.pi * r**2
print("Circumference: " + str(C))
print("Area: " + str(A))

# calculations with lists: python doesn't know how to perform calculations on lists..
height = [1.73, 1.68, 1.71, 1.89, 1.79]
weight = [65.4, 59.2, 63.6, 88.4, 68.7]
weight / height ** 2
# .. use arrays!!


# 4. Numpy arrays
# ---------------
# numpy arrays can only have one type & have their own methods / behavior

baseball = [180, 215, 210, 210, 188, 176, 209, 200]
# create a numpy array from baseball..
np_baseball = np.array(baseball)
np_height_m = np.array(height) * 0.0254
np_weight_kg = np.array(weight) * 0.453592
bmi = np_weight_kg / np_height_m ** 2
print(bmi)

# subsetting..
light = bmi < 15000
print(bmi[light])

# 2D arrays..
baseball = [[180, 78.4],
            [215, 102.7],
            [210, 98.5],
            [188, 75.2]]
np_baseball = np.array(baseball)
# print out the 4th row of np_baseball..
print(np_baseball[3,:])
# select the entire second column of np_baseball..
np_weight = np_baseball[:,1]
# array additions..
conversion = np.array([0.0254,0.453592])
np_baseball * conversion
# 180*0.0254
# 78.4*0.453592

# generate random data..
height = np.round(np.random.normal(1.75, 0.2, 5000),2)
weight = np.round(np.random.normal(60.32, 15, 5000),2)
np_city = np.column_stack((height, weight))
# some stats..
np.mean(np_baseball[:,0])
np.median(np_baseball[:,0])
np.corrcoef(np_baseball[:,0], np_baseball[:,1])
np.std(np_baseball[:,0])












