# Importing Data into Python (Part 1)
# -----------------------------------



# reading a text file..
filename = 'data/seaslug.txt'
file = open(filename, mode='r') # 'r' / 'w' is read / write
text = file.read()
file.close()

# context managers with..
with open('data/seaslug.txt','r') as file:
    print(file.read())
# ..




















