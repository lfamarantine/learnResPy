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
# expect if we collected the same data again, to draw actionable conclusions
# from data and to draw more general conclusions from relatively few data observations..

# random number generators & hacker statistics..

# simulate probability of 4 successive heads in coin-flips..
np.random.seed(42)
rn = np.random.random(size=4)
heads = rn < 0.5
np.sum(heads)

# generalize..
n_all_heads = 0 # initialise
for _ in range(10000):
    heads = np.random.random(size=4) < 0.5
    n_heads = np.sum(heads)
    if n_heads == 4:
        n_all_heads += 1

n_all_heads / 10000

# alternatively: sampling probability distributions..
samples = np.random.binomial(4, 0.5, 10000)

# examples of poisson-processes: the next event is independent of the previous one..
# - natural births at a given hospital
# - hits on a website during a given hour
# - meteor strikes
# - aviation incidents
# - buses in poissonville

# properties:
# - 1 parameter: The number or r arrivals of a Poisson process in a given time interval with average rate of
# lambda arrivals per interval is Poisson distributed.
# - limit of poisson distribution for low probability of success and large number of trials (eg. for rare events)






