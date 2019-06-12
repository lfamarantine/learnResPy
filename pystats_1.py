# Statistical Thinking in Python I
# --------------------------------
import pandas as pd
import numpy as np

# 1. Graphical exploratory data analysis
# --------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
df_swing = pd.read_csv('data/pystats1_2008_swing_states.csv')

# generating a histogram & using seaborn styling..
sns.set()
_ = plt.hist(df_swing['dem_share']['dem_share'], bins=20)
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
# 1. binomial distribution
# 2. poisson distribution
# 3. normal distribution
# 4. exponential distribution

# binomial distribution..
# ---
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

# poisson distribution..
# ---
# examples of poisson-processes: the next event is independent of the previous one..
# - natural births at a given hospital
# - hits on a website during a given hour
# - meteor strikes
# - aviation incidents
# - buses in poissonville
samples_poisson = np.random.poisson(10, 10000)
# example: 251/115 no-hitters per season..
n_nohitters = np.random.poisson(251/115, 10000)
# probability of having >=7 no-hitters?
n_large = np.sum(n_nohitters>=7)
p_large = n_large / 10000
print(p_large)

# properties:
# - 1 parameter: The number or r arrivals of a Poisson process in a given time interval with average rate of
# lambda arrivals per interval is Poisson distributed.
# - limit of poisson distribution for low probability of success and large number of trials (eg. for rare events)

# normal distribution..
# ---
# normal PDF..
# 3 random distributions..
samples_std1 = np.random.normal(20, 1, 100000)
samples_std3 = np.random.normal(20, 3, 100000)
samples_std10 = np.random.normal(20, 10, 100000)
# histograms..
plt.hist(samples_std1, bins=100, normed=True, histtype='step')
plt.hist(samples_std3, bins=100, normed=True, histtype='step')
plt.hist(samples_std10, bins=100, normed=True, histtype='step')
# legend..
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'))
plt.ylim(-0.01, 0.42)
plt.show()

# normal CDF..
x_std1, y_std1 = ecdf(samples_std1)
x_std3, y_std3 = ecdf(samples_std3)
x_std10, y_std10 = ecdf(samples_std10)
# plot CDFs..
_ = plt.plot(x_std1, y_std1, marker='.', linestyle='none')
_ = plt.plot(x_std3, y_std3, marker='.', linestyle='none')
_ = plt.plot(x_std10, y_std10, marker='.', linestyle='none')
# legend..
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'), loc='lower right')

# exponential distribution
# ---
# Rare events can be modelled with a Poisson process, and the waiting time between arrivals of a Poisson process is
# exponentially distributed. Exponential distribution has a single parameter, which is tau, the typical interval time.
# Example: What is the total waiting time for the arrival of two different Poisson processes?
def successive_poisson(tau1, tau2, size=1):
    """Compute time for arrival of 2 successive Poisson processes."""
    # draw samples out of first exponential distribution..
    t1 = np.random.exponential(tau1, size)
    # draw samples out of second exponential distribution..
    t2 = np.random.exponential(tau2, size)
    return t1 + t2

# mean waiting time for a no-hitter / hitting of the cycle: 764 / 715
waiting_times = successive_poisson(764, 715, 100000)
_ = plt.hist(waiting_times, bins=100, normed=True, histtype='step')
_ = plt.xlabel('waiting time')
_ = plt.ylabel('probability')


















