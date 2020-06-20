# Python Data Science Toolbox (Part 3)
# ------------------------------------
from datetime import date
import datetime
import numpy as np

# core framework ------------------------------------------------------------------------------------------------------
# define a class with methods
class person:
    # self references current instance of the class, used to access variables of the class
    def __init__(self, name, age, city):
        self.name = name
        self.age = age
        self.city = city

    # add a method to the class
    def msg(self):
        print("My name is " + self.name + ", I am " + str(self.age) + " years old and live in " + self.city + ".")

    # add a method with parameters
    def born(self, days = 365):
        """Returns the year the person was born.
        Parameters:
            n:      age, int
                    the age of the person
            days:   days, int, default 365
                    days in year
        """
        d = datetime.date.today() - datetime.timedelta(days=(self.age*days))
        z = d.timetuple().tm_year
        print("Max was born in " + str(z), ".")

# create a class
p = person("Max", 32, "Berlin")
p.msg()
p.born()


# key things to keep in mind ------------------------------------------------------------------------------------------
a = np.array([1, 2, 3])
# state <-> attributes (eg. a.shape is an attribute)
# behaviour <-> methods (eg. a.reshape(3, 1) is a method)
# list all methods and attributes of an object: dir(a)
# a method is a function within a class


# adding methods to a class
class dog:

    def type(self, name):
        print("The dog is a " + name)
# note:
# - self is always the 1st argument
# what for is self?
# class are objects, how does one refer to data in a particular object?
# self is a stand-in for a particular object in a class

class employee:

    def set_name(self, x):
        self.name = x

    def set_salary(self, x):
        self.salary = x

    def raise_salary(self, x):
        self.salary = self.salary + x

emp = employee()
emp.set_name('Max King')
emp.set_salary(50000)
print(emp.salary)
# change value manually
emp.salary = emp.salary + 1500
# or use method
emp.raise_salary(200)
emp.salary



