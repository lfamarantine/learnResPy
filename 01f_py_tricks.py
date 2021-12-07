# ---------------   Useful Tricks from the Book 'Python Tricks' by Dan Bader   ---------------
import numpy as np
import math

# ----- optional function arguments
def foo(x, *args, **kwargs):

    print(x)

    if args:
        print(args)

    if kwargs:
        print(kwargs)


foo('hello', 1, 2, 3, 'red', col_2='blue')
foo('hello', 1, 2, 3, col_1='red', col_2='blue')


# ----- printing vectors
def print_vector(x, y, z):
    print('<%s, %s, %s>' % (x, y, z))


print_vector(1, 2, 3)


# ----- every Class needs a __repr__
class Car:
    def __init__(self, color, mileage):
        self.color = color
        self.mileage = mileage


Car('bmw', 'blue')
print(Car('bmw', 'blue'))


class Car:
    def __init__(self, color, mileage):
        self.color = color
        self.mileage = mileage

    def __str__(self):
        return f'a {self.color} car'

    def __repr__(self):
        # !r conversion flag to make sure the output string uses repr(self.color)
        return f'{self.__class__.__name__}(' f'{self.color!r}, {self.mileage})'


print(Car('bmw', 'blue')) # str
Car('bmw', 'blue') # repr




# ----- lambda function evaluated immediately
# -- double
(lambda x, y: (x + y)**2)(2, 3)
# -- vector
x_1 = np.arange(1, 11)
y_1 = np.arange(11, 21)
(lambda x, y: (x + y)**2)(x_1, y_1)


# ----- decorators
from functools import wraps

def strong(func):
    @wraps(func) # optional but recommended: to copy meta data from inner function to outer
    def wrapper():
        return '<strong>' + func() + '</strong>'
    return wrapper

def emphasis(func):
    @wraps(func)
    def wrapper():
        return '<em>' + func() + '</em>'
    return wrapper

@strong
@emphasis
def greet():
    return 'Hello!'


greet()
help(greet)



# ----- Instance, Class, and Static Methods Demystified
# - generic
class MyClass:
    # - instance method can modify class + object state
    def method(self):
        return ('instance method called', self)

    # - independent method, works like a simple function
    # - used to signal to the developer the class design
    @staticmethod
    def staticmethod(): # can't modify any object
        return 'static method called'

    # - allow you to formulate alternative constructors for your class (since only one __init__ possible)
    # - can't modify object instance but class state
    @classmethod
    def classmethod(cls):
        return ('class method called', cls)


MyClass().method()
MyClass().staticmethod()
MyClass().classmethod()

# - practical example
class Pizza:
    def __init__(self, ingredients):
        self.ingredients = ingredients

    def __repr__(self):
        return f'Pizza({self.ingredients!r})'

    @classmethod
    def margherita(cls):
        return cls(['mozzarella', 'tomato'])

    @classmethod
    def prosciutto(cls):
        return cls(['mozzarella', 'tomato', 'ham'])

    @staticmethod
    def circle_area(radius):
        return radius ** 2 * math.pi


Pizza.margherita()
Pizza.prosciutto()
Pizza.circle_area(4)


# ----- Exploring modules / objects
# find all classes/functions of an object
import datetime
dir(datetime)
dir(datetime.date)
# find a specific object (eg. date)
xi = [i for i in dir(datetime) if "date" in i.lower()]


# ----- Install venv from terminal:
# 1. python3 -m venv ./venv
# 2. source ./venv/bin/activate # to explicitly active venv & tell pip where to install pkg's


# ----- Generators
# definition: A generator is a much more convenient way of writing an iterator
# example:
def countdown(n):
    print("Counting down from", n)
    while n > 0:
        yield n
        n -= 1


k = countdown(4)
k # <generator object countdown at 0x0000022C5197DD60>
k.__next__() # Counting down from 4 <4>
k.__next__() # Counting down from 3 <3>


# ----- Assertions
example = [5, 3, 1, 6, 6]
booleans = [False, False, True, False]
any(example) # True > all numbers above 0 are ‘Truthy’
any(booleans) # True
all(example) # True
all(booleans) # False


