# import matplotlib.pyplot as plt
# import pandas as pd
#
# # Loading the CSV file
# file = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\varying\\"
# our_approach = file + 'outdoor_air_temp.csv'
# df = pd.read_csv(our_approach)
#
# # Calculating first order differences
# df['Newcastle_diff'] = df['Newcastle'].diff()
# df['BOPTEST_diff1'] = df['BOPTEST_1'].diff()
# df['BOPTEST_diff2'] = df['BOPTEST_2'].diff()
#
# # Extracting original and difference data for boxplot
# Newcastle_original = df['Newcastle'].dropna()
# BOPTEST_original_1 = df['BOPTEST_1'].dropna()
# BOPTEST_original_2 = df['BOPTEST_2'].dropna()
# Newcastle_diff = df['Newcastle_diff'].dropna()
# BOPTEST_diff1 = df['BOPTEST_diff1'].dropna()
# BOPTEST_diff2 = df['BOPTEST_diff2'].dropna()
#
# # Data for plots
# original_data = [BOPTEST_original_1, BOPTEST_original_2]
# diff_data = [BOPTEST_diff1, BOPTEST_diff2]
#
# # Labels for each dataset
# labels = ['Test Case 1 - Heating scenario','Test Case 2 - Cooling dominated scenario']
#
# # Setting up the figure with two rows of subplots
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), constrained_layout=True)
#
# # Original data subplot
# ax1.boxplot(original_data, vert=True, patch_artist=True, labels=labels)
# ax1.set_title('Comparative boxplot of ambient temperature')
# ax1.set_ylabel('Temperature (°C)')
# ax1.set_xlabel('(a)')
# ax1.set_xticklabels([])  # Disable x-axis labels for ax3
# ax1.yaxis.grid(True)  # Adding grid for better readability
#
# # Colors for the box plots
# colors = ['pink', 'lightblue','green']
# for patch, color in zip(ax1.patches, colors):
#     patch.set_facecolor(color)
#
# # First order differences subplot
# ax2.boxplot(diff_data, vert=True, patch_artist=True, labels=labels)
# ax2.set_title('Analysis of temperature change dynamics: First-order differences')
# ax2.set_ylabel('Temperature change (°C)')
# ax2.set_xlabel('(b)')
# ax2.yaxis.grid(True)
#
# # Coloring the box plots
# for patch, color in zip(ax2.patches, colors):
#     patch.set_facecolor(color)
# plt.savefig('C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\varying\\outdoor.png')
# plt.show()

import matplotlib.pyplot as plt
import pandas as pd

# Loading the CSV file
file = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\varying\\"
our_approach = file + 'outdoor_air_temp.csv'
df = pd.read_csv(our_approach)

# Calculating first order differences
df['Newcastle_diff'] = df['Newcastle'].diff()
df['BOPTEST_diff1'] = df['BOPTEST_1'].diff()
df['BOPTEST_diff2'] = df['BOPTEST_2'].diff()

# Extracting original and difference data for boxplot
Newcastle_original = df['Newcastle'].dropna()
BOPTEST_original_1 = df['BOPTEST_1'].dropna()
BOPTEST_original_2 = df['BOPTEST_2'].dropna()
Newcastle_diff = df['Newcastle_diff'].dropna()
BOPTEST_diff1 = df['BOPTEST_diff1'].dropna()
BOPTEST_diff2 = df['BOPTEST_diff2'].dropna()

# Data for plots
original_data = [BOPTEST_original_1, BOPTEST_original_2]
diff_data = [BOPTEST_diff1, BOPTEST_diff2]

# Labels for each dataset
labels = ['Test Case 1 - Heating scenario','Test Case 2 - Cooling dominated scenario']

# Setting up the figure with two rows of subplots
fig, ax1 = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)

# Original data subplot
ax1.boxplot(original_data, vert=True, patch_artist=True, labels=labels)
ax1.set_title('Comparative boxplot of ambient temperature')
ax1.set_ylabel('Temperature (°C)')
# ax1.set_xlabel('(a)')
# ax1.set_xticklabels([])  # Disable x-axis labels for ax3
ax1.yaxis.grid(True)  # Adding grid for better readability

# Colors for the box plots
colors = ['pink', 'lightblue','green']
for patch, color in zip(ax1.patches, colors):
    patch.set_facecolor(color)

# # First order differences subplot
# ax2.boxplot(diff_data, vert=True, patch_artist=True, labels=labels)
# ax2.set_title('Analysis of temperature change dynamics: First-order differences')
# ax2.set_ylabel('Temperature change (°C)')
# ax2.set_xlabel('(b)')
# ax2.yaxis.grid(True)
#
# # Coloring the box plots
# for patch, color in zip(ax2.patches, colors):
#     patch.set_facecolor(color)
plt.savefig('C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\varying\\outdoor.png')
plt.show()