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
def strong(func):
    def wrapper():
        return '<strong>' + func() + '</strong>'
    return wrapper

def emphasis(func):
    def wrapper():
        return '<em>' + func() + '</em>'
    return wrapper

@strong
@emphasis
def greet():
    return 'Hello!'


greet()



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

# install venv from terminal:
# 1. python3 -m venv ./venv
# 2. source ./venv/bin/activate # to explicitly active venv & tell pip where to install pkg's



