# Python Data Science Toolbox (Part 3)
# ------------------------------------
from datetime import date
import datetime
import numpy as np
import pandas as pd

# core framework of classes in py -------------------------------------------------------------------------------------
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


# first principles ----------------------------------------------------------------------------------------------------
a = np.array([1, 2, 3])
# - state <-> attributes (eg. a.shape is an attribute)
# - behaviour <-> methods (eg. a.reshape(3, 1) is a method)
# - list all methods and attributes of an object: dir(a)
# - a method is a function within a class


# simplest example for adding methods to a class
class dog:

    def type(self, name):
        print("The dog is a " + name)

# note:
# - self is always the 1st argument
# what for is self?
# class are objects, how does one refer to data in a particular object?
# self is a stand-in for a particular object in a class

class employee:
    # attribute name
    def set_name(self, x):
        self.name = x
    # attribute salary
    def set_salary(self, x):
        self.salary = x
    # method raise_salary
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

# attributes can be set all at once and don't need all the code above for every attribute:
# - __init__() method is called every time an object is created
# - use __init__() to set define all attributes at once in the constructor
# - use camelcase for classes, lower_snake_case for functions/attributes

class employee:

    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

emp = employee("Max King", 30000)
emp.name
emp.salary



# inheritance, polymorphism & encapsulation ---------------------------------------------------------------------------

# more on inheritance (extending functionality of existing code), polymorphism (creating unified interface) &
# encapsulation (bundling of data methods)

# class methods ----------------------------
# - classmethod
# - staticmethod
# Main use of class methods is defining methods that return an instance of the class
# but aren't using the same code as __init__(). An example..

class EnhDate:

    def __init__(self, year, month, day):
        self.year, self.month, self.day = year, month, day

    # Define a class method from_str
    @classmethod
    def from_str(cls, datestr):
        # split the string at "-"
        parts = datestr.split("-")
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
        # return class instance
        return cls(year, month, day)

bd = EnhDate.from_str('2020-04-30')
print(bd.year)
print(bd.month)
print(bd.day)


# inheritance ----------------------------
# check whether it's a child of a class: isinstance(MyClass)

class Employee:
    min_salary = 10000

    def __init__(self, name, salary=min_salary):
        self.name = name
        if salary >= Employee.min_salary:
            self.salary = salary
        else:
            self.salary = Employee.min_salary

# a new class Manager inheriting from Employee
class Manager(Employee):
    pass

# the new inherited class with its own objects
class Manager(Employee):
  def display(self):
    print("Manager ", self.name)

# define a new Manager object
mng = Manager("Max King", 70000)
mng.name
mng.display()
mng.salary
mng.min_salary


# adding additional attributes to a parents class attributes
class Employee:
    def __init__(self, name, salary=30000):
        self.name = name
        self.salary = salary

    def give_raise(self, amount):
        self.salary += amount

# add Manager class with additional project attribute
class Manager(Employee):
    # add a constructor
    def __init__(self, name, salary=50000, project=None):
        # call the parent's constructor
        Employee.__init__(self, name, salary)
        # assign project attribute
        self.project = project

    def display(self):
        print("Manager ", self.name)


# modifying a pandas dataframe class
# define ModDF inherited from pd.DataFrame and add new attribute
class ModDF(pd.DataFrame):

    def __init__(self, *args, **kwargs):
        pd.DataFrame.__init__(self, *args, **kwargs)
        self.created_on = datetime.date.today()


mdf = ModDF({"a": [1, 2], "b": [3, 4]})
print(mdf.values)
print(mdf.created_on)


# operators in classes ----------------------------
# let's say you want to compare two instances of a class that have the same data:
e1 = Employee("Max King", 100)
e2 = Employee("Max King", 100)
# they are not the same..
e1==e2
# this is because py compares the references, not the actual data
# to circumvent this, use __eq__ and other operators in the class

# methods:
# - __eq__ for equal
# - __ne__ for not equal
# - __ge__ for >=
# - __le__ for <=
# - __gt__ for >
# - __lt__ for <
# there is also __hash__ method, that allows equal objects to be treated the same


class Employee:
    def __init__(self, name, salary=30000):
        self.name = name
        self.salary = salary

    def __eq__(self, other):
        return (self.salary == other.salary and self.name == other.name)

e1 = Employee("Max King", 100)
e2 = Employee("Max King", 100)
e1==e2
