# Python ABC
Python learning material with exercises based on various sources.

#| topic	| progress
-| ------ | --------
1a | intro to python	| complete
1b | python data science toolbox (part I) | complete
1c | python data science toolbox (part II)	| complete
1d | python data science toolbox (part III)	| complete
1e | python classes	| complete
1f | python tricks	| in progress
2a | importing data in python (part I)	| complete
2b | importing data in python (part II)	| complete
3a | cleaning data in python	| complete
3b | pandas foundations	| complete
3c | manipulating dataframes with pandas | complete	
3d | merging dataframes with pandas	| complete
4a | introduction to data visualization with python | in progress	
4b | interactive data visualization with bokeh | in progress
4c | python & excel | complete
5a | statistical thinking in python (part I) | complete
5b | statistical thinking in python (part II) | complete
5c | linear algebra in python | in progress
5d | basics on optimisation problems | in progress
6a | supervised learning with scikit-learn	| complete	
6b | machine learning with tree-based models | complete
6c | unsupervised learning in python	| complete
6d | deep learning in python | complete
6e | machine learning with the experts: kaggle school budgets | in progress
6f | machine learning algo concept illustrations | in progress
7a | statistics revision | complete
8a | useful python commands | in progress
8b | aws setup | in progress
8c | shell operations | in progress
8d | third-party data downloads | in progress
9a | package dev in py | complete



Useful Data Science links
-------------------------

- machine learning: https://pythonprogramminglanguage.com/python-machine-learning/


Key Commands in Python Terminal
-------------------------------

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


Key Commands in Python Script Editor
------------------------------------

- code completition: Preferences| Editor | General | Code Completition for some settings and ```Ctrl+Space``` 
for code completition after calling a library for example (eg. pd. ```Ctrl+Space``` )

