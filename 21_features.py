# Commands to Remember in Python
# ------------------------------
import pandas as pd
import numpy as np


# ---------------------------------------------
# PYTHON TERMINAL OPERATIONS
# ---------------------------------------------
# .. useful operations performed in terminal

# -- installing packages: python -m pip install pandas
# -- specific version installation: python -m pip install pandas==0.24.1
# -- find package version: pip show pandas
# -- alter code execution command (default is shift+alt+e): File -> Settings -> Keymap ->
#    Other -> Execute selection in console
# -- manage package dependencies:
#       1. create a requirements.txt file in root directory of project by: pip freeze > requirements.txt
#       2. after pull, create new empty environ incl. global site-packages (delete old env sub-folders)
#       3. run pip install -r requirements.txt


# ---------------------------------------------
# PYTHON CONSOLE OPERATIONS FOR SETUP
# ---------------------------------------------
# -- working directory:
import os
os.getcwd() # get directory
os.chdir('/Users/dariopopadic/PycharmProjects/learnResPy/') # set directory



# ---------------------------------------------
# DATA MANIPULATION
# ---------------------------------------------
# -- dynamically change DF names with variables
for i in range(5):
    vars()['df_' +str(i)] = pd.DataFrame(np.random.rand(10, 3), columns=list('abc'))


# -- dataframe key operations
# .. modifying a column based on another:
df = pd.DataFrame({'x0': np.random.normal(2, 4, 10), 'y0': np.random.normal(-2, 4, 10)})
df.x0[df.y0 < 0] += 2








