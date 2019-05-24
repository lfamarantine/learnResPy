# Statistical Thinking in Python
# ------------------------------
import pandas as pd
import numpy as np

# 1. Graphical exploratory data analysis
# --------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
df_swing = pd.read_csv('data/pystats1_2008_swing_states.csv')

# generating a histogram & using seaborn styling..
sns.set()
_ = plt.hist(df_swing['dem_share'], bins=20)
_ = plt.xlabel('percent of vote for Obama')
_ = plt.ylabel('number of counties')

# ..issues with histograms: binning bias, not plotting all data -> use bee swarm plot

# bee swarm plot..
_ = sns.swarmplot(x='state', y='dem_share', data=df_swing)
_ = plt.xlabel('state')
_ = plt.ylabel('percent of vote for Obama')

# plotting empirical cumulative distribution functions (ECDF)..
x = np.sort(df_swing['dem_share'])
y = np.arange(1, len(x)+1) / len(x)
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('percent of vote for Obama')
_ = plt.ylabel('ECDF')
plt.margins(0.02) # keeps data off plot edges

# multiple ECDF's in one plot..
# defining a ecdf-function..
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, len(data)+1) / n

    return x, y


# 2. Quantitative exploratory data analysis
# -----------------------------------------
df_all_states = pd.read_csv('data/pystats1_2008_all_states.csv')
# percentiles..
np.percentile(df_swing['dem_share'], [25,50,75])
# boxplots..
_ = sns.boxplot(x='east_west', y='dem_share', data=df_all_states)
_ = plt.xlabel('region')
_ = plt.ylabel('percent of vote for Obama')

# variance & stdev..
np.var(df_swing['dem_share'])
np.sqrt(np.var(df_swing['dem_share']))
np.std(df_swing['dem_share'])

# covariance & pearson correlation coefficient..
# scatter-plot..
_ = plt.plot(df_all_states['total_votes']/1000, df_all_states['dem_share'], marker='.', linestyle='none')
_ = plt.xlabel('total votes (thousands)')
_ = plt.ylabel('percent of vote for Obama')

# covariance  matrix..
cov = np.cov(df_all_states['total_votes'].iloc[:10], df_all_states['dem_share'].iloc[:10])
cov


# 3. Probabilistic logic & statistical inference
# ----------------------------------------------
# statistical inference: to draw probabilistic conclusions about what we might
# expect if we collected the same data again, to ddraw actionable conclusions
# from data and to draw more general conclusions from relatively few data observations..

# random number generators & hacker statistics..












