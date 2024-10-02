import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Function to calculate alpha(t) based on the formula
def calculate_alpha(t_out, t_ref, beta, delta):
    return 1 / (1 + np.exp(beta * (np.abs(t_out - t_ref) - delta)))


# Parameters
out_temp = np.linspace(-10, 20, 500)
out_temp_summer = np.linspace(-30, 30, 500)
beta_winter = -1
beta_summer = 1
ref_day = 22.5 * np.ones(500)
ref_winter_night = 15 * np.ones(500)
ref_night_summer = 30 * np.ones(500)
delta_values =[17.5] # [0, 17.5, 35]
colors = ['cyan']#['green', 'cyan', 'blue']  # Colors for different delta values


def alpha(outdoor_temp, ref, beta, delta):
    return 1 / (1 + np.exp(beta * (abs(outdoor_temp-ref) - delta)))


fig = plt.figure(figsize=(20, 8))
gs = gridspec.GridSpec(1, 2, figure=fig)
# fig.suptitle(r'Dynamic thermal weight $\alpha(t)$ V.S. fixed thermal weight $\alpha$=0.5',y=0.915)  # Set a title for the whole figure


fig = plt.figure(figsize=(30, 23))
gs = gridspec.GridSpec(1, 2, figure=fig)

# Adjust the spacing between subplots
plt.subplots_adjust(left=0.5, right=0.99, top=0.99, bottom=0.7, hspace=0.8, wspace=0.4)

# Calculate the x-axis values as the absolute difference between outdoor temperature and reference temperature
x_values_winter_day = np.abs(out_temp - ref_day)
x_values_winter_night = np.abs(out_temp - ref_winter_night)
x_values_summer_day = np.abs(out_temp_summer - ref_day)
x_values_summer_night = np.abs(out_temp_summer - ref_night_summer)

# Plot for winter season

# Calculate the x-axis values as the absolute difference between outdoor temperature and reference temperature
x_values_winter_day = np.abs(out_temp - ref_day)
x_values_winter_night = np.abs(out_temp - ref_winter_night)
x_values_summer_day = np.abs(out_temp_summer - ref_day)
x_values_summer_night = np.abs(out_temp_summer - ref_night_summer)

# Plot for winter season
ax0 = fig.add_subplot(gs[0, 0])
for delta, color in zip(delta_values, colors):
    alpha_winter_day = alpha(out_temp, ref_day, beta_winter, delta)
    alpha_winter_night = alpha(out_temp, ref_winter_night, beta_winter, delta)
    ax0.plot(x_values_winter_day, alpha_winter_day, label=r'$\alpha(t)$ for winter $(\delta = {}$) - occupied hours'.format(delta), color=color, linestyle='-', linewidth=4)
    # ax0.plot(x_values_winter_night, alpha_winter_night, label=r'$\alpha(t)$ for winter $(\delta = {}$) - unoccupied hours'.format(delta), color=color, linestyle='--', linewidth=1)

# Titles and labels
ax0.set_title(r'Thermal weight $\alpha(t)$ for different $\delta$ values in winter seasons', fontsize=18)
ax0.set_xlabel(r'The absolute difference $|t_{out} - t_{ref}|$', fontsize=14)
ax0.set_ylabel(r'$\alpha(t)$', fontsize=20)
ax0.legend(fontsize=12)
ax0.grid(True)

# Plot for summer season
ax01 = fig.add_subplot(gs[0, 1])
for delta, color in zip(delta_values, colors):
    alpha_winter_day = alpha(out_temp, ref_day, beta_winter, delta)
    alpha_winter_night = alpha(out_temp, ref_winter_night, beta_winter, delta)
    # ax01.plot(x_values_winter_day, alpha_winter_day,label=r'$\alpha(t)$ for winter $(\delta = {}$) - occupied hours'.format(delta), color=color, linestyle='-',linewidth=4)
    ax01.plot(x_values_winter_night, alpha_winter_night,label=r'$\alpha(t)$ for winter $(\delta = {}$) - unoccupied hours'.format(delta), color=color,linestyle='--', linewidth=1)

# Titles and labels
ax01.set_title(r'Thermal weight $\alpha(t)$ for different $\delta$ values in summer seasons', fontsize=18)
ax01.set_xlabel(r'The absolute difference $|t_{out} - t_{ref}|$', fontsize=14)
ax01.set_ylabel(r'$\alpha(t)$', fontsize=20)
ax01.legend(fontsize=12)
ax01.grid(True)




plt.tight_layout()
# plt.savefig('C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\varying\\delta_visual.png')
plt.show()
