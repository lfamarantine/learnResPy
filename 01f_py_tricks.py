# ---------------   Useful Tricks from the Book 'Python Tricks' by Dan Bader   ---------------
import numpy as np

# optional function arguments
def foo(x, *args, **kwargs):

    print(x)

    if args:
        print(args)

    if kwargs:
        print(kwargs)


foo('hello', 1, 2, 3, 'red', col_2='blue')
foo('hello', 1, 2, 3, col_1='red', col_2='blue')


# printing vectors
def print_vector(x, y, z):
    print('<%s, %s, %s>' % (x, y, z))


print_vector(1, 2, 3)


# every Class needs a __repr__
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


Car('bmw', 'blue') # repr
print(Car('bmw', 'blue')) # print



# lambda function evaluated immediately
# -- double
(lambda x, y: (x + y)**2)(2, 3)
# -- vector
x_1 = np.arange(1, 11)
y_1 = np.arange(11, 21)
(lambda x, y: (x + y)**2)(x_1, y_1)

# decorators
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