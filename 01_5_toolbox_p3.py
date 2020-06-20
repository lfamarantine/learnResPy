# Python Data Science Toolbox (Part 3)
# ------------------------------------
from datetime import date
import datetime

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


