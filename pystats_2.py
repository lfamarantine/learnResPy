# Statistical Thinking in Python II
# ---------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Parameter estimation by optimization
# ---------------------------------------
np.random.seed(42)
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, len(data)+1) / n

    return x, y

# how often do we get no hitters?
# ---
nohitter_times = np.array([ 843, 1613, 1101,  215,  684,  814,  278,  324,  161,  219,  545,
        715,  966,  624,   29,  450,  107,   20,   91, 1325,  124, 1468,
        104, 1309,  429,   62, 1878, 1104,  123,  251,   93,  188,  983,
        166,   96,  702,   23,  524,   26,  299,   59,   39,   12,    2,
        308, 1114,  813,  887,  645, 2088,   42, 2090,   11,  886, 1665,
       1084, 2900, 2432,  750, 4021, 1070, 1765, 1322,   26,  548, 1525,
         77, 2181, 2752,  127, 2147,  211,   41, 1575,  151,  479,  697,
        557, 2267,  542,  392,   73,  603,  233,  255,  528,  397, 1529,
       1023, 1194,  462,  583,   37,  943,  996,  480, 1497,  717,  224,
        219, 1531,  498,   44,  288,  267,  600,   52,  269, 1086,  386,
        176, 2199,  216,   54,  675, 1243,  463,  650,  171,  327,  110,
        774,  509,    8,  197,  136,   12, 1124,   64,  380,  811,  232,
        192,  731,  715,  226,  605,  539, 1491,  323,  240,  179,  702,
        156,   82, 1397,  354,  778,  603, 1001,  385,  986,  203,  149,
        576,  445,  180, 1403,  252,  675, 1351, 2983, 1568,   45,  899,
       3260, 1025,   31,  100, 2055, 4043,   79,  238, 3931, 2351,  595,
        110,  215,    0,  563,  206,  660,  242,  577,  179,  157,  192,
        192, 1848,  792, 1693,   55,  388,  225, 1134, 1172, 1555,   31,
       1582, 1044,  378, 1687, 2915,  280,  765, 2819,  511, 1521,  745,
       2491,  580, 2072, 6450,  578,  745, 1075, 1103, 1549, 1520,  138,
       1202,  296,  277,  351,  391,  950,  459,   62, 1056, 1128,  139,
        420,   87,   71,  814,  603, 1349,  162, 1027,  783,  326,  101,
        876,  381,  905,  156,  419,  239,  119,  129,  467])
# compute mean no-hitter time tau..
tau = np.mean(nohitter_times)
# draw out of an exponential distribution with parameter tau..
inter_nohitter_time = np.random.exponential(tau, 100000)
# plot the PDF..
_ = plt.hist(inter_nohitter_time, bins=50, normed=True, histtype='step')
_ = plt.xlabel('Games between no-hitters')
_ = plt.ylabel('PDF')

# does the data fit our story?
# ---
x, y = ecdf(nohitter_times)
x_theor, y_theor = ecdf(inter_nohitter_time)
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker=".", linestyle='none')
plt.margins(0.02)
plt.xlabel('Games between no-hitters')
plt.ylabel('CDF')

# linear regression by least squares..
# ---
df_swing = pd.read_csv('data/pystats1_2008_swing_states.csv')
x = df_swing['dem_share']
y = df_swing['total_votes']
slope, intercept = np.polyfit(x, y, 1)
print(slope)
print(intercept)

# example linear regression..
df = pd.read_csv('data/pystats2_female_literacy_fertility.csv')
illiteracy = df['female literacy']
fertility = df['fertility']

_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')
plt.margins(0.02)
_ = plt.xlabel('percent illiterate')
_ = plt.ylabel('fertility')
# linear regression..
a, b = np.polyfit(illiteracy, fertility, 1)
print('slope =', a, 'children per woman / percent illiterate')
print('intercept =', b, 'children per woman')
# theoretical line to plot..
x = np.array([0, 100])
y = x * a + b
# add regression line to plot
_ = plt.plot(x, y)

# optimum visualisation..
# ---
# specify slopes to consider..
a_vals = np.linspace(0, 0.1, 200)
# initialize sum of square of residuals..
rss = np.empty_like(a_vals)
# compute sum of square of residuals for each value..
for i, a in enumerate(a_vals):
    rss[i] = np.sum((fertility - a*illiteracy - b)**2)

plt.plot(a_vals, rss, '-')
plt.xlabel('slope (children per woman / percent illiterate)')
plt.ylabel('sum of square of residuals')

# anscombe's quartet plot's lesson: always explore the data first visually!


# 2. Generating bootstrap replicates
# ----------------------------------
np.random.choice([1,2,3,4,5], size=5)
# draw a sample from data with replacement..
df_sample = np.random.choice(nohitter_times, size=100)
# Terminology: If we have a data set with n repeated measurements, a 'bootstrap sample' is an array of length n that
# was drawn from the original data with replacement. A 'bootstrap replicate' is a single value of a statistic
# computed from a bootstrap sample.

# bootstrapping by hand: How many unique bootstrap samples can be drawn from [-1, 0 , 1]?
print('there are', 3 ** 3, 'unique samples, and the maximum mean is 1.')

# visualizing bootstrap samples..
# ---
rainfall_tmp = pd.read_csv('data/pystats2_sheffield_weather_station.csv', delimiter=r"\s+", skiprows=8)
rainfall = rainfall_tmp['rain']

for i in range(50):
    # generate bootstrap sample..
    bs_sample = np.random.choice(rainfall, size=len(rainfall))
    # compute and plot ECDF from bootstrap sample
    x, y = ecdf(bs_sample)
    _ = plt.plot(x, y, marker='.', linestyle='none', color='gray', alpha=0.1)

# compute and plot ECDF from original data
x, y = ecdf(rainfall)
_ = plt.plot(x, y, marker='.')
plt.margins(0.02)
_ = plt.xlabel('yearly rainfall (mm)')
_ = plt.ylabel('ECDF')

# bootstrap replicate function..
def bootstrap_replicate_1d(data, func):
    """Generate bootstrap replicate of 1D data."""
    bs_sample = np.random.choice(data, len(data))
    return func(bs_sample)

# .. many boostrap replicates..
bs_replicates = np.empty(10000)
for i in range(10000):
    bs_replicates[i] = bootstrap_replicate_1d(rainfall, np.mean)

_ = plt.hist(bs_replicates, bins=30, normed=True)
_ = plt.xlabel('mean rainfall')
_ = plt.ylabel('PDF')

# generalised..
def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""
    # initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)
    # generate replicates
    for i in range(size):
        bs_replicates[i] =  bootstrap_replicate_1d(data, func)
    return bs_replicates

# example..
bs_replicates = draw_bs_reps(rainfall, np.mean, size=10000)
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel('mean annual rainfall (mm)')
_ = plt.ylabel('PDF')
# confidence interval..
np.percentile(bs_replicates, [2.5, 97.5])

# pairs bootstrap..
# ---
df_swing = pd.read_csv('data/pystats1_2008_swing_states.csv')
dem_share = df_swing['dem_share']
total_votes = df_swing['total_votes']

inds = np.arange(len(total_votes))
bs_inds = np.random.choice(inds, len(inds))
bs_total_votes = total_votes[bs_inds]
bs_dem_share = dem_share[bs_inds]
bs_slope, bs_intercept = np.polyfit(bs_total_votes, bs_dem_share, 1)
print(bs_slope, bs_intercept)

# general function..
def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""
    # set up array of indices to sample from
    inds = np.arange(len(x))
    # initialize replicates
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)
    # generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)
    return bs_slope_reps, bs_intercept_reps

# example..
bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(illiteracy, fertility, 1000)
plt.hist(bs_slope_reps, bins=50, normed=True)


# 3. Introduction to hypothesis testing
# -------------------------------------

# generating a permutation sample..
# ---
def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""
    # concatenate the data sets: data
    data = np.concatenate([data1, data2])
    # permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)
    # split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]
    return perm_sample_1, perm_sample_2


# hypothesis testing..
# ---
# alternative formulation: null hypothesis significance testing (NHST)
# statistical significance =! practical significance
# p-value: the probability of observing a test statistic as extreme ore more extreme than the one you observed, given
# that the null hypothesis is true.

# generating permutation replicates..
def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""
    # initialize array of replicates..
    perm_replicates = np.empty(size)
    for i in range(size):
        # generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)
        # compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)

    return perm_replicates


# bootstrap hypothesis testing..
# ---
force_a = np.array([1.612, 0.605, 0.327, 0.946, 0.541, 1.539, 0.529, 0.628, 1.453,
                    0.297, 0.703, 0.269, 0.751, 0.245, 1.182, 0.515, 0.435, 0.383,
                    0.457, 0.73 ])
force_b = np.array([0.172, 0.142, 0.037, 0.453, 0.355, 0.022, 0.502, 0.273, 0.72 ,
                    0.582, 0.198, 0.198, 0.597, 0.516, 0.815, 0.402, 0.605, 0.711,
                    0.614, 0.468])

# Another juvenile frog was studied, Frog C, and you want to see if Frog B and Frog C have similar impact forces.
# Unfortunately, you do not have Frog C's impact forces available, but you know they have a mean of 0.55 N. Because
# you don't have the original data, you cannot do a permutation test, and you cannot assess the hypothesis that the
# forces from Frog B and Frog C come from the same distribution. You will therefore test another, less restrictive
# hypothesis: The mean strike force of Frog B is equal to that of Frog C.
translated_force_b = force_b - np.mean(force_b) + 0.55
# take bootstrap replicates of Frog B's translated impact forces..
bs_replicates = draw_bs_reps(translated_force_b, np.mean, 10000)
# compute fraction of replicates that are less than the observed Frog B force
p = np.sum(bs_replicates <= np.mean(force_b)) / 10000
# p-value
print('p = ', p)


# 3. Hypothesis testing examples
# ------------------------------
# process: 1. define H0, 2. figure out how to simulate it, 3. define what it means to be more extreme 4. p-value

















