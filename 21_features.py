# Nice Features with Python
# -------------------------
import pandas as pd
import numpy as np

# -- dynamically change DF names with variables
for i in range(5):
    vars()['df_' +str(i)] = pd.DataFrame(np.random.rand(10, 3), columns=list('abc'))










