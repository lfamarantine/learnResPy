# Python Data Science Toolbox (Part 1)
# ------------------------------------
import pandas as pd
import builtins
from functools import reduce

# 1. Writing your own functions
# -----------------------------

# function w\o parameter..
def square(): # .. function header
    new_value = 4 ** 2
    print(new_value)
# call the function..
square()

# function with parameter & return..
# docstrings -> describe what the function does.. in """ """
def square(value):
    """ Return the square of a value. """
    new_value = value ** 2
    return new_value
num = square(4)

# remember: y2 = print(2) -> assigning a variable y2 to a function that prints a value but does not return a value will
# result in that variable y2 being of type NoneType

# tuples
# ------

# tuples (return multiple values): like a list, immutable, constructed using parentheses
even_nums = (2, 4, 6)
# unpack tuple into several values..
a, b, c = even_nums
print(a)
# access tuple elements..
even_nums[0]

# functions with multiple arguments & return values..
def shout_all(word1, word2):
    shout1 = word1 + "!!!"
    shout2 = word2 + "!!!"
    # Construct a tuple with shout1 and shout2..
    shout_words = (shout1, shout2)
    return shout_words
# calling the function & storing values..
yell1, yell2 = shout_all('congratulations', 'you')

# case study..
# build a dictionary in which the keys are the names of languages and the values are
# the number of tweets in the given language..
df = pd.read_csv('data/tweets.csv')
langs_count = {}
col = df['lang']
for entry in col:
    # if the language is in langs_count, add 1..
    if entry in langs_count.keys():
        langs_count[entry] += 1
    # else add the language to langs_count, set the value to 1..
    else:
        langs_count[entry] = 1
print(langs_count)


# 2. Default arguments, variable-length arguments & scope
# -------------------------------------------------------

# scope: part of the program where an object or name may be accessible
# 3 types of scope: 1. global | 2. local | 3. built-in scope (names in the pre-defined built-ins module - eg. print())
# priority: 1. local -> 2. global -> 3. built-in

# local scope..
new_val = 10
def square(value):
    """Returns the square number."""
    new_val = value ** 2
    return new_val

print(square(10))
print(new_val)


# global vs local scope..
new_val = 10
def square(value):
    """Returns the square number."""
    global new_val
    new_val = new_val ** 2
    return new_val

print(square(3))
print(new_val)

# builtins..
dir(builtins)

# nested functions..
# why? .. repeated calculations within a function
def mod2plus5(x1, x2, x3):
    """Returns the remainder plus 5 of three values."""
    def inner(x):
        """Returns the remainder plus 5 of a value."""
        return x % 2 + 5
    return (inner(x1), inner(x2), inner(x3))

mod2plus5(1, 2, 3)

# other use case of nested functions..
def raise_val(n):
    """Returns the inner function."""
    def inner(x):
        """Raise x to the power of n."""
        raised = x ** n
        return raised
    return inner

square = raise_val(2)
cube = raise_val(3)
print(square(2), cube(4))
# closure: inner function remembers the state of its enclosing scope when called

# using nonlocal..
def outer():
    """Prints the value of n."""
    n = 1
    def inner():
        nonlocal n
        n = 2
        print(n)
    inner()
    print(n)

outer()

# -> LEGB rule: local, enclosing, global, built-in

# default arguments for functions..
def power(number, pow=1):
    """Raise the number to the power of pow."""
    new_value = number ** pow
    return new_value

power(9, 2)
power(9)

# flexible arguments..
def add_all(*args):
    """Sum of all values in *args together."""
    sum_all = 0
    for num in args:
        sum_all += num
    return sum_all

add_all(1, 2, 5)
add_all(1, 2)

def print_all(**kwargs):
    """Print out key-value pairs in **kwargs."""
    for key, value in kwargs.items():
        print(key + ": " + value)

print_all(name="dumbledore", job="headmaster")


# 3. Lambda-functions & error-handling
# ------------------------------------

# lambda functions allow to write functions quickly..
raise_to_power = lambda x, y: x ** y
raise_to_power(2, 3)
# anonymous functions..
# function map takes 2 arguments: map(func, seq)
# map applies function to all elements in the sequence
nums = [48, 6, 9, 21, 1]
square_all = map(lambda num: num ** 2, nums)
square_all
print(list(square_all))

# filter lambda functions..
fellowship = ['frodo', 'samwise', 'merry', 'pippin', 'aragorn', 'boromir', 'legolas', 'gimli', 'gandalf']
result = filter(lambda member: len(member) > 6, fellowship)
result_list = list(result)
print(result_list)

# reduce lambda functions..
stark = ['robb', 'sansa', 'arya', 'brandon', 'rickon']
result = reduce(lambda item1, item2: item1 + item2, stark) # ..  returns a single value as a result
print(result)


# introduction to error handling..
def sqrt(x):
    """Returns a square root of a number."""
    try:
        return x ** 0.5
    except:
        print("x must be an int or float.")

sqrt(4)
sqrt("hi")

# only use type-errors & other exceptions..
def sqrt(x):
    """Returns a square root of a number."""
    try:
        return x ** 0.5
    except TypeError:
        print("x must be an int or float.")
# handling exceptions: not allowing for negative numbers..
def sqrt(x):
    """Returns a square root of a number."""
    if(x < 0):
        raise ValueError("x must be non-negative")
    try:
        return x ** 0.5
    except TypeError:
        print("x must be an int or float.")

sqrt(-4)











