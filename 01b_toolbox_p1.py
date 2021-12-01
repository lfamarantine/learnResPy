wwww
# ------------------------------------
# Content:
# - numpy array operations
# - plotting (time-series, scatter-plot, histogram etc.) & plotting methods
# - dictionary operations
# - pandas introduction (creating pd tables, subsetting with loc/iloc)
# - boolean operators (and, or etc.)
# - if-else operations, for- & while-loops
# - random number generation


# 1. Matplotlib
# -------------
import pkg_resources
import matplotlib.pyplot as plt
pkg_resources.get_distribution("matplotlib").version
import numpy as np
import pandas as pd

# generate random series..
pop_tmp = np.random.uniform(1,100,150)
pop = np.cumsum(pop_tmp)
year = np.array(range(1950, 2100))
len(pop)
len(year)
# time-series chart..
plt.plot(year, pop)
plt.show()

# A. scatterplot..
income = np.random.uniform(0, 100000, 150)
plt.scatter(income, pop)
plt.xscale('log')

# B. histograms..
help(plt.hist)
plt.hist(income, bins=50)

# C. customization: labels, ticks, bubbles, colors, text..
gdp_cap = abs(np.random.normal(120000,300000,len(pop)))
life_exp = abs(np.random.normal(70,10,len(pop)))
plt.scatter(gdp_cap, life_exp)
# labels..
plt.xlabel("GDP per Capita [in USD]")
plt.ylabel("Life Expectancy [in years]")
plt.title("World Development in 2007")
# ticks..
tick_val = [1000,10000,100000]
tick_lab = ['1k','10k','100k']
plt.xticks(tick_val, tick_lab)
# bubbles..
plt.scatter(gdp_cap, life_exp, s=pop)
# create random colors..
color = list()
color_def = ["yellow","red","blue","green","black"]
for i in range(0,150):
    color.append(np.random.choice(color_def))
print(color[1])
len(color)
plt.scatter(x = gdp_cap, y = life_exp, s = np.array(pop) * 0.1, c = color, alpha = 0.8)
# text..
plt.text(1550, 71, 'India')
plt.scatter(x = gdp_cap, y = life_exp, s = np.array(pop) * 0.1, c = color, alpha = 0.8)


# 2. Dictionaries & Pandas
# ------------------------

# motivation..
countries = ['spain', 'france', 'germany', 'norway']
capitals = ['madrid', 'paris', 'berlin', 'oslo']
ind_ger = countries.index("germany")
print(capitals[ind_ger])
# .. not very elegant way..

# create dictionary..
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' }
print(europe.keys()) # .. keys have to be immutable objects
print(europe['norway'])
print('norway' in europe)
# modifying dictionaries..
# .. add italy to europe
europe['italy'] = 'rome'
print(europe)
# .. remove italy from europe
del(europe['italy'])
print(europe)

# dictionary of dictionaries..
europe = { 'spain': { 'capital':'madrid', 'population':46.77 },
           'france': { 'capital':'paris', 'population':66.03 },
           'germany': { 'capital':'berlin', 'population':80.62 },
           'norway': { 'capital':'oslo', 'population':5.084 } }

print(europe['france']['capital'])
print(europe['france']['population'])

# create sub-dictionary data..
data = {
    'capital':'rome',
    'population':59.83
}
# add data to europe..
europe['italy'] = data
print(europe)


# pandas..
# 2D numpy array only takes 1 data type
# -> pandas is a high level data manipulation tool using dataframes

# read a csv file..
brics = pd.read_csv("data/brics.csv", index_col=0)

# build dataframe from dictionary..
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr = [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]
dict = { 'country':names, 'drives_right':dr, 'cars_per_cap':cpc }
print(dict)
cars = pd.DataFrame(dict)
print(cars)
# new row_labels..
row_labels = ['US', 'AUS', 'JAP', 'IN', 'RU', 'MOR', 'EG']
cars.index = row_labels
print(cars)

# pandas brackets..
cars['country'] # type(cars['country']) -> series (1D labeled array)
cars[['country']] # type(cars[['country']]) -> dataframe

# get 2nd & 3rd row..
cars[1:3]
cars[1:] # get from 2nd row
# .. [] works, but only offers limited functionality..
# to get my_array[rows, columns] functionality: loc & iloc
# loc: label-based
# iloc: integer position-based
cars.loc[["JAP","MOR"]]
cars.loc[["JAP","MOR"],["country","cars_per_cap"]]
cars.loc[:,["country","cars_per_cap"]]

cars.iloc[[2,5]]
cars.iloc[[2,5],[0,2]]
cars.iloc[:,[0,2]]


# 3. Logic, Control Flow & Filtering
# ----------------------------------

# compare arrays..
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])
print(my_house >= 18)
print(my_house < your_house)

# boolean operators: and, or, not..
y = 5
y > 3 and y < 13
y < 7 or y > 13
not True
not False

# with numpy arrays..
np.logical_and(y>3, y<13)
np.logical_or(y<7, y>13)
print(np.logical_and(my_house < 11, your_house < 11))
# select respective fields..
my_house[np.logical_and(my_house>15, my_house<=19)]

# if, else, elif
# .. if / else / elif also possible as standalone
z = 3
if z % 2 == 0:
    print("z is divisible by 2")
elif z % 3 == 0:
    print("z is divisible by 3")
else:
    print("z is neither divisible by 2 nor by 3")

# filtering pandas dataframe..
is_huge = brics["area"] > 8
brics[is_huge]
# .. or directly..
brics[brics["area"] > 8]
# .. boolean operators..
brics[np.logical_and(brics["area"]>8, brics["area"]<10)]


# 4. Loops
# --------

# while loop..
# .. used for convergence problems such as minimization/maximization
offset = -6
while offset != 0 :
    print("correcting...")
    if offset > 0:
        offset = offset - 1
    else:
        offset = offset + 1
    print(offset)

# for loop..
cpc = [809, 731, 588, 18, 200, 70, 45]
type(cpc)
for i in cpc:
    print(cpc.index(i))
    print(i)

# for loop with index..
for index, i in enumerate(cpc):
    print("index " + str(index) + ":" + str(i))

# for loop over string..
for i in "family":
    print(i.capitalize())

# loop over list of lists..
house = [["hallway", 11.25],
         ["kitchen", 18.0],
         ["living room", 20.0],
         ["bedroom", 10.75],
         ["bathroom", 9.50]]
for i in house:
    print("the " + str(i[0]) + " is " + str(i[1]) + " sqm")

# looping data structures (dictionaries, numpy arrays or pandas dataframes)..
# dictionaries..
world = {"afghanistan": 30.55,
         "albania": 2.77,
         "algeria": 39.21}
for key, i in world.items():
    print(key + " -- " + str(i))
# .. dictionaries are unordered

# numpy array..
height = np.array([1.73, 1.68, 1.71, 1.89, 1.79])
weight = np.array([65.4, 59.2, 63.6, 88.4, 68.7 ])
meas = np.array([height, weight])

# 1d array..
for i in height:
    print(i)
# 2d array..
for i in np.nditer(meas):
    print(i)

# pandas dataframes..
brics
for lab, row in brics.iterrows():
    print(lab + " : " + row["capital"])

# add column..
for lab, row in brics.iterrows():
    brics.loc[lab, "name_length"] = len(row["country"])
# with apply..
brics["name_length2"] = brics["country"].apply(len)
brics["name2"] = brics["country"].apply(str.upper)


# 5. Case Study: Hacker Statistics..
# ----------------------------------
# same random numbers..
np.random.seed(123)

coin = np.random.randint(0, 2) # randomly generate 0 or 1
print(np.random.randint(1, 7)) # randomly simulate a dice

step = 50
# roll the dice..
dice = np.random.randint(1,7)
if dice <= 2:
    step = step - 1
elif (dice <= 5):
    step = step + 1
else:
    step = step + np.random.randint(1,7)

# Print out dice and step
print(dice)
print(step)


# toss game..
outcomes = []
for x in range(10):
    coin = np.random.randint(0,2)
    if coin == 0:
        outcomes.append("heads")
    else:
        outcomes.append("tails")
    print(outcomes)
    print(x)

# random walk of steps..
np.random.seed(123)
random_walk = [0]
for x in range(100):
    step = random_walk[-1] # last value
    dice = np.random.randint(1,7)

    if dice <= 2:
        step = max(0, step - 1) # ..can't go below 0
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)
    print(step)
    random_walk.append(step)

# plot random_walk..
plt.plot(random_walk)

# what's the chance that you reach 60 steps high?
# 100 runs of 10 tosses..
final_tails = []
for x in range(1000):
    tails = [0]
    for x in range(10):
        coin = np.random.randint(0, 2)
        tails.append(tails[x] + coin)
    final_tails.append(tails[-1])
    print(final_tails)

plt.hist(final_tails, bins=10)
plt.show()


# simulation: 250 runs of steps problem..
all_walks = []
for i in range(250):
    random_walk = [0]
    for x in range(100):
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)

        # Implement clumsiness -> 0.1% chance of falling down
        if np.random.rand(1) <= 0.001:
            step = 0

        random_walk.append(step)
    all_walks.append(random_walk)

np_aw_t = np.transpose(np.array(all_walks))
# path..
plt.plot(np_aw_t)
# distribution (last step)..
ends = np_aw_t[-1]
plt.hist(ends)
# probabilty for reaching 60 steps high..
print(len(ends[ends >= 60]) / len(ends))








