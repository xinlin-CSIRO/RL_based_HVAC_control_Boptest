import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
file = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\varying\\"
our_approach = file + 'Newcastle_site_gym_testresults.csv' #
df = pd.read_csv(our_approach)

head = int(6 * 24 * 5)
end = int(6 * 24 * (5 + 5))
indoor_our_approach = df['indoor_temp'].values[head:end] + 0.25
up_boundary = df['boundary_1'].values[head:end]
low_boundary = df['boundary_0'].values[head:end]
outdoor = df['outdoor_air'].values[head:end]

# Initialize figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15), constrained_layout=True)

# Plot indoor temperature with boundaries
ax1.plot(indoor_our_approach, 'cyan', label='Our method')
ax1.plot(up_boundary, 'g:', label='Upper boundary')
ax1.plot(low_boundary, 'g:', label='Lower boundary')
ax1.set_ylabel('Temperature (°C)')
ax1.legend(loc="lower right")
ax1.set_xticks([])  # Hide x-axis ticks for clarity on ax1

# Plot outdoor temperature
ax2.plot(outdoor, 'navy', label='Outdoor air temperature')
ax2.set_ylabel('Temperature (°C)')
ax2.legend(loc="lower right")

# Define specific dates and their indices
specific_dates = ['May 23', 'May 24', 'May 25', 'May 26', 'May 27','May 28']
index_dates = [i * 24 * 6 for i in range(len(specific_dates))]  # Correcting the multiplier
ax2.set_xticks(index_dates)
ax2.set_xticklabels(specific_dates)

plt.show()
