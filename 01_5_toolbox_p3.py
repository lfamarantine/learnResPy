# Python Data Science Toolbox (Part 3)
# ------------------------------------

# create a class with methods

class person:
    # self references current instance of the class, used to access variables of the class
    def __init__(self, name, age, city):
        self.name = name
        self.age = age
        self.city = city
    # add a method to the class
    def msg(self):
        print("My name is " + self.name + ", I am " + str(self.age) + " years old and live in " + self.city + ".")

p = person("Max", 36, "Berlin")
p.msg()


