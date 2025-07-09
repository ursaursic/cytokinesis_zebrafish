# created August 2024 by Alison
# very similar code to ingression velocity measurement
# used to track the retraction of one end of the contractile band in M-phase
# code was written with the help of chatGPT:
# OpenAI. (2024, August 15). ChatGPT (GPT-4) [Large language model]. OpenAI. https://www.openai.com/


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress


#BOOTSTRAPPING YAY
def weighted_bootstrap(data, func=np.mean, weights=None, n_resamples=1000, rng=None):
    '''
    from ursa
    '''

    # Create a default random number generator if none is provided
    rng = np.random.default_rng() if rng is None else rng

    # If no weights are provided, use equal weights
    if weights is None:
        weights = np.ones(len(data))

    # Generate bootstrap samples by randomly selecting elements with replacement,
    # using the provided weights as sampling probabilities
    bootstrap_samples = rng.choice(
        data,
        size=(n_resamples, len(data)),
        replace=True,
        p=weights / np.sum(weights),  # Normalize weights to sum to 1
        axis=0
    )

    # Apply the statistic function to each resample along the axis of resampling
    statistic = func(bootstrap_samples, axis=1)

    # Compute the 95% confidence interval (2.5th and 97.5th percentiles)
    ci = np.percentile(statistic, [2.5, 97.5], axis=0)

    return np.mean(statistic, axis=0), ci


# loading multiple files with different time intervals
files = [
    # '/Volumes/cytokinesis-zebrafish/Processed data/band_length/1_SNT_measurements/20210625_CSV_Properties_10x_to66.csv',
    '/Volumes/cytokinesis-zebrafish/Processed data/band_length/1_SNT_measurements/20250703_CSV_Properties_10x1.csv',
    '/Volumes/cytokinesis-zebrafish/Processed data/band_length/1_SNT_measurements/20250703_CSV_Properties_10x2.csv',
    '/Volumes/cytokinesis-zebrafish/Processed data/band_length/1_SNT_measurements/20250703_CSV_Properties_10x3_embryo1.csv',
    '/Volumes/cytokinesis-zebrafish/Processed data/band_length/1_SNT_measurements/20250703_CSV_Properties_10x3_embryo2.csv'
]
intervals = [60,60,60,60]
# multi_factor = [2,2,2,2]
M_phase_sync = [14,17,16,15] # in minutes, synchronise slice data
# pixel_conversion = [0.39]

# Define lists to store all data points
all_ys = []
all_slices = []

# shades_of_grey = ['#C2BCC7', '#C2BCC7', '#C2BCC7', '#C2BCC7', '#C2BCC7']

data_all = []
data_time_all = []

#open files
for i in range(len(files)):
    file = files[i]
    frame_to_min = intervals[i] / 60
    m_phase_offset = M_phase_sync[i]  # get M-phase sync offset in minutes
    # pixel_to_micron = pixel_conversion[i]
    data = pd.read_csv(file)

    # Set scale
    data['Y'] = data['PathLength']
    # data['Slice'] = data['PathID'] * frame_to_min
    data['Slice'] = data['PathID'] * frame_to_min - m_phase_offset + 17

    offset = 17 - M_phase_sync[i]


    data_all.append(data['Y'].values[offset:-(17-offset)])
    data_time_all.append([i for i in range(len(data['Slice'].values[offset:-(17-offset)]))])


    # Store ingression data
    all_ys.append(data['Y'])
    all_slices.append(data['Slice'])

print(data_all)


# Find common time range
common_slices = np.unique(np.concatenate(all_slices))
#
# Interpolate mean ingression speed to match common time range
mean_retraction = np.zeros_like(common_slices)
for ys, slices in zip(all_ys, all_slices):
    mean_retraction += np.interp(common_slices, slices, ys)

mean_retraction /= len(files)

# print(mean_retraction)

# EDIT: calculating the standard deviation doesn't make so much sense, because the y-values are normalised to end in the same point
#  Calculate standard deviation
# std_deviation = np.zeros_like(common_slices)
# for ys, slices in zip(all_ys, all_slices):
#     std_deviation += np.interp(common_slices, slices, ys) ** 2
# 
# std_deviation = np.sqrt(std_deviation / len(files) - mean_retraction ** 2)

#---- Ursas edit
# std_deviation = np.zeros_like(common_slices)
# for ys, slices in zip(all_ys, all_slices):
#     std_deviation += (np.interp(common_slices, slices, ys) - mean_retraction) ** 2
#
# std_deviation /= len(files)
#
# print(std_deviation)

#FITTING
# Filter the data for the time range between minute 1 and minute 4 (to measure retraction velocity before it gets resuced)
# mask = (common_slices >= 0) & (common_slices <= 5)
# filtered_slices = common_slices[mask]
# filtered_mean_retraction = mean_retraction[mask]
#
# # Fit a linear regression to the mean ingression data
# slope, intercept, r_value, p_value, std_err = linregress(filtered_slices, filtered_mean_retraction)
#
# # Calculate residuals
# predicted_retraction = intercept + slope * filtered_slices
# residuals = filtered_mean_retraction - predicted_retraction
#
# # Calculate standard deviation of residuals (Residual Standard Error)
# residual_std_error = np.sqrt(np.sum(residuals**2) / (len(filtered_slices) - 2))
#
# # Calculate 95% confidence interval for the slope
# confidence_interval = 1.96 * std_err

#PRINTING
# Print results
# print(f"Slope: {slope:.3f} µm/min")
# print(f"Standard Error of the Slope: {std_err:.3f} µm/min")
# print(f"95% Confidence Interval for the Slope: [{slope - confidence_interval:.3f}, {slope + confidence_interval:.3f}] µm/min")
# print(f"Residual Standard Error: {residual_std_error:.3f} µm")


#PLOTTING
# line for M-phase
plt.axvline(x=17, label='M-phase start', color='k', linestyle='--')
plt.axvline(x=30, label='approx. interphase start', color='grey', linestyle='--')

# Plot individual datasets
for ys, slices in zip(all_ys, all_slices):
    plt.plot(slices, ys, lw=2, color='k',alpha=0.2)


# Plot mean ingression speed
plt.plot(common_slices, mean_retraction, lw=4, color='#FF9100', label='Mean band length')


# Plot fitted line
# plt.plot(common_slices, intercept + slope * common_slices, 'b--', label=f'Fit: {slope:.3f} µm/min', lw=2)

# Plot standard deviation
# plt.fill_between(common_slices, mean_retraction - std_deviation, mean_retraction + std_deviation, color='grey', alpha=0.2, label='stdev')

# Labeling axes and plot title
plt.xlabel('Time (min)', fontsize=14)
plt.ylabel('Band length (µm)', fontsize=14)
plt.tick_params(axis='both', direction='in', labelsize=12, length=6, width=1)

plt.legend(fontsize=12)

plt.show()

# save_results_to = '/Volumes/cytokinesis-zebrafish/Processed data/band_length/4_plots/'
# plt.savefig(save_results_to + 'band-length-3d_M-align_interphase.png', format='png')

plt.plot(data_time_all, data_all, 'k.')
plt.show()
