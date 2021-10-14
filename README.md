# Python Basics
Python basics material.

#| topic | 
-| ------ | 
1a | intro to python	| 
1b | python data science toolbox (part I) | 
1c | python data science toolbox (part II)	| 
1d | python data science toolbox (part III)	| 
1e | python classes	| 
1f | python tricks	|
2a | importing data in python (part I)	| 
2b | importing data in python (part II)	| 
3a | cleaning data in python	| 
3b | pandas foundations	| 
3c | manipulating dataframes with pandas | 	
3d | merging dataframes with pandas	| 
4a | interactive data visualization with bokeh | 
4b | python & excel | 
5a | statistical thinking in python (part I) | 
5b | statistical thinking in python (part II) | 
5c | linear algebra in python | 
5d | basics on optimisation problems | 
6a | supervised learning with scikit-learn	| 	
6b | machine learning with tree-based models | 
6c | unsupervised learning in python	| 
6d | deep learning in python | 
6e | machine learning with the experts: kaggle school budgets | 
6f | machine learning algo concept illustrations | 
7a | statistics revision | 
8a | useful python commands | 
8b | aws setup | 
8c | shell operations |
8d | third-party data downloads | 
9a | package dev in py |



Useful Data Science links
-------------------------

- machine learning: https://pythonprogramminglanguage.com/python-machine-learning/


Key Commands in Py Terminal
----------------------------

- installing packages: ```python -m pip install pandas```

- specific version installation: ```python -m pip install pandas==0.24.1```

- find package version: ```pip show pandas```

- alter code execution command (default is shift+alt+e): 
```File -> Settings -> Keymap -> Other -> Execute selection in console```

- manage package dependencies:
      1. create a requirements.txt file in root directory of project by: ```pip freeze > requirements.txt```
      2. after pull, create new empty environ incl. global site-packages (delete old env sub-folders)
      3. run ```pip install -r requirements.txt```

- install py-package from github: ```pip install git+https://github.com/adamhajari/spyre.git```


Key Commands in Py Script Editor
---------------------------------

- code completition: Preferences| Editor | General | Code Completition for some settings and ```Ctrl+Space``` 
for code completition after calling a library for example (eg. pd. ```Ctrl+Space``` )

