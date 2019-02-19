# Python Data Science Toolbox (Part 1)
# ------------------------------------

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








