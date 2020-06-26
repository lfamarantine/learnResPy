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
# a class can only have one __init__ method and if one wants to define alternative constructors, @classmethod
# is used
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

    # compare keys and class of keys too
    def __eq__(self, other):
        return (self.salary == other.salary and self.name == other.name and type(self) == type(other))

e1 = Employee("Max King", 100)
e2 = Employee("Max King", 100)
e1==e2

# class referencing --------------------------------
# what happens when an object is compared to an object of a child class? child's classed is called
# printable representation of a class by using one of:
# 1) __str__ (informal)
# 2) __repr__ (for developers)

# an example..

class Customer:

    def __init__(self, name, balance):
        self.name, self.balance = name, balance

    def __str__(self):
        cust_str = """
        Customer: 
            name: {name}
            balance: {balance}
        """.format(name = self.name, \
                   balance = self.balance)
        return cust_str

c_0 = Customer("Max King", 1200)
print(c_0)


class Customer:

    def __init__(self, name, balance):
        self.name, self.balance = name, balance

    def __repr__(self):
        return "Customer('{name}', {balance}).format(name = self.name, balance = self.balance)"

c_0 = Customer("Max King", 1200)
print(c_0)


# exceptions ----------------------------------------

# try - except - finally
# an example..
def cust_inv(x, ind):
  try:
    return 1/x[ind]
  except ZeroDivisionError:
    print("Cannot divide by zero!")
  except IndexError:
    print("Index out of range!")

a = [0, 3, 5, 9]
print(cust_inv(a, 1))
print(cust_inv(a, 0))
print(cust_inv(a, 4))



# other useful points ---------------------------------
# when to use inheritance?
# as a rule of thumb, if the class hierarchy violates the liskov of substitution principle, one should
# not be using inheritance -> no lsp = no inheritance
# liskov substitution principle: wherever employee works, manager should work too
# - all classes are public in py!


# data access management via:
# 1) naming conventions:
#       starts with a single _ (internal, not part of public API) -> obj._att_name, obj._method_name
#       starts with a double __ (internal, closest to private class) -> obj.__att_name, obj.__method_name used mainly
#       to prevent name clashes from inherited classes. So only use to prevent attributes being inherited.
# 2) @property to customise access (to control attribute access)
#       use "protected" attribute with leading _ to store data

# an example
class Customer:
    def __init__(self, name, new_bal):
        self.name = name
        if new_bal < 0:
            raise ValueError("Invalid balance!")
        self._balance = new_bal

    @property
    def balance(self):
        return self._balance

    # add a setter balance() method
    @balance.setter
    def balance(self, new_bal):
        # validate
        if new_bal < 0:
            raise ValueError("Invalid balance!")
        self._balance = new_bal
        print("Setter method called")

cust = Customer("Max King", 30000)
cust.balance = 3000
print(cust.balance)


# read-only "created_at" attribute for the inherited pd class example from above..

class ModDF(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        pd.DataFrame.__init__(self, *args, **kwargs)
        self._created_at = datetime.date.today()

    def to_csv(self, *args, **kwargs):
        temp = self.copy()
        temp["created_at"] = self._created_at
        pd.DataFrame.to_csv(temp, *args, **kwargs)

    @property
    def created_at(self):
        return self._created_at


df = ModDF({"col1": [1, 2], "col2": [3, 4]})

try:
    df.created_at = '2035-07-13'
except AttributeError:
    print("Could not set attribute")





