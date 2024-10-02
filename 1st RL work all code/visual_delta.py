import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')



# Assuming the file path and file reading parts remain unchanged
file = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\varying\\"
our_approach = file + 'delta_comparsion_2.xlsx'
data_source=pd.ExcelFile(our_approach)
our_method_heat = pd.read_excel(data_source, 'Heat_delta_17')
head=9600
end=len(our_method_heat)-1
best_heat_indoor_alpt_t=np.array(our_method_heat['indoor_temp'][head:end]).astype(float)
# best_heat_indoor_alpt_t = resampled_data = np.array([np.mean(best_heat_indoor_alpt_t[i:i+4]) for i in range(0, len(best_heat_indoor_alpt_t), 4)])[4:]

thermal_kpi_heat = np.array(our_method_heat['themal_kpi'][head:end]).astype(float)
outdoor_heat = np.array(our_method_heat['outdoor_air'][head:end]).astype(float)
energy_kpi_heat = np.array(our_method_heat['energy_kpi'][head:end]).astype(float)
weight_heat = np.array(our_method_heat['weight'][head:end]).astype(float)

#####Bestest heat benchmark###0######################
nest_heat_benchmark = pd.read_excel(data_source, 'Heat_delta_0')
head2=head
end2=end
best_heat_indoor_delta_0=np.array(nest_heat_benchmark['indoor temp'][head2:end2]).astype(float)
# outdoor_2 = np.array(nest_heat_benchmark['outdoor air'][head2:end2]).astype(float)
up_boundary_heat = np.array(nest_heat_benchmark['boundary_1'][head2:end2]).astype(float)
low_boundary_heat = np.array(nest_heat_benchmark['boundary_0'][head2:end2]).astype(float)
thermal_kpi_heat_benchmark_delta_0 = np.array(nest_heat_benchmark['themal kpi'][head2:end2]).astype(float)
energy_kpi_heat_benchmark_delta_0 = np.array(nest_heat_benchmark['energy kpi'][head2:end2]).astype(float)
weight_heat_benchmark_delta_0 = np.array(nest_heat_benchmark['weight'][head2:end2]).astype(float)

#####Bestest heat benchmark###35######################
hest_heat_benchmark_35 = pd.read_excel(data_source, 'Heat_delta_35')
head2=head
end2=end
best_heat_indoor_delta_35=np.array(hest_heat_benchmark_35['indoor temp'][head2:end2]).astype(float)
thermal_kpi_heat_benchmark_delta_35 = np.array(hest_heat_benchmark_35['themal kpi'][head2:end2]).astype(float)
energy_kpi_heat_benchmark_delta_35 = np.array(hest_heat_benchmark_35['energy kpi'][head2:end2]).astype(float)
weight_heat_benchmark_delta_35 = np.array(hest_heat_benchmark_35['weight'][head2:end2]).astype(float)



our_method_cool = pd.read_excel(data_source, 'Air_delta_17')
head=2400
end=len(our_method_cool)-1
best_cool_indoor_alpt_t=np.array(our_method_cool['indoor_temp'][head:end]).astype(float)
thermal_kpi_cool= np.array(our_method_cool['themal_kpi'][head:end]).astype(float)
outdoor_cool = np.array(our_method_cool['outdoor_air'][head:end]).astype(float)
energy_kpi_cool = np.array(our_method_cool['energy_kpi'][head:end]).astype(float)
weight_cool = np.array(our_method_cool['weight'][head:end]).astype(float)
up_boundary_cool = np.array(our_method_cool['boundary_1'][head:end]).astype(float)
low_boundary_cool = np.array(our_method_cool['boundary_0'][head:end]).astype(float)
#####Bestest cool benchmark###0######################
Cool_benchmark_1 = pd.read_excel(data_source, 'Air_delta_0')
best_cool_indoor_delta_0=np.array(Cool_benchmark_1['indoor_temp'][head:end]).astype(float)
thermal_kpi_cool_benchmark_delta_0 = np.array(Cool_benchmark_1['themal_kpi'][head:end]).astype(float)
energy_kpi_cool_benchmark_delta_0 = np.array(Cool_benchmark_1['energy_kpi'][head:end]).astype(float)
weight_cool_benchmark_delta_0 = np.array(Cool_benchmark_1['weight'][head:end]).astype(float)

#####Bestest cool benchmark###35######################
cool_benchmark_2 = pd.read_excel(data_source, 'Air_delta_35')
best_cool_indoor_delta_35=np.array(cool_benchmark_2['indoor_temp'][head:end]).astype(float)
thermal_kpi_cool_benchmark_delta_35 = np.array(cool_benchmark_2['themal_kpi'][head:end]).astype(float)
energy_kpi_cool_benchmark_delta_35 = np.array(cool_benchmark_2['energy_kpi'][head:end]).astype(float)
weight_cool_benchmark_delta_35 = np.array(cool_benchmark_2['weight'][head:end]).astype(float)






print('done')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Create a figure with GridSpec layout
fig = plt.figure(figsize=(30, 19))
gs = gridspec.GridSpec(3, 4, figure=fig)
# fig.suptitle(r'Dynamic thermal weight $\alpha(t)$ V.S. fixed thermal weight $\alpha$=0.5',y=0.915)  # Set a title for the whole figure

ax1 = fig.add_subplot(gs[0:1, 0:2])  # This adds a subplot that spans the first row entirely
# ax1.set_title('Best Hydronic Heat Pump: Peak heat days', fontsize=17)
ax1.set_title('Test Case 1 - Heating scenario', fontsize=24)

ax1.plot(best_heat_indoor_delta_0, label=r'Indoor temperature - $\delta=0 $', color='green', linewidth=2)
ax1.plot(best_heat_indoor_alpt_t, label=r'Indoor temperature - $\delta=17.5 $', color='cyan', linewidth=2)
ax1.plot(best_heat_indoor_delta_35, label=r'Indoor temperature - $\delta=35 $', color='blue', linewidth=2)
ax1.plot(up_boundary_heat, 'g:')
ax1.plot(low_boundary_heat, 'g:')
ax1.legend(loc="lower right", fontsize=16)
# specific_dates = ['Jan 17', 'Jan 19', 'Jan 21', 'Jan 23', 'Jan 25', 'Jan 27', 'Jan 29', 'Jan 31']
# index_dates = [i * 96 * 2 for i in range(len(specific_dates))]  # Each day has 96 steps (4 steps per hour * 24 hours)
# ax1.set_xticks(index_dates)
ax1.set_xticklabels([])
ax1.set_ylabel('Temperature (°C)', fontsize=16)

ax2 = fig.add_subplot(gs[0:1, 2:4])  # This adds a subplot that spans the first row entirely
ax2.set_title('Test Case 2 - Cooling dominated scenario', fontsize=24)

ax2.plot(best_cool_indoor_delta_0, label=r'Indoor temperature - $\delta=0$', color='green', linewidth=2)
ax2.plot(best_cool_indoor_alpt_t, label=r'Indoor temperature - $\delta=17.5$', color='cyan', linewidth=2)
ax2.plot(best_cool_indoor_delta_35, label=r'Indoor temperature - $\delta=35$', color='blue', linewidth=2)
ax2.plot(up_boundary_cool, 'g:')
ax2.plot(low_boundary_cool, 'g:')
ax2.legend(loc="lower right", fontsize=16)
ax2.set_xticklabels([])
ax2.set_ylabel('Temperature (°C)', fontsize=16)

# # 2nd row
# ax3 = fig.add_subplot(gs[1, 0:2])  # This adds a subplot that spans the first row entirely
# ax3.set_title(r'Outdoor temperature', fontsize=12)
# ax3.plot(outdoor_heat, label=r'Outdoor temperature', color='black', linewidth=2)
# ax3.legend(loc="lower right")
# ax3.set_ylabel('Temperature (°C)', fontsize=12)
# ax3.set_xticklabels([])
#
# ax4 = fig.add_subplot(gs[1, 2:4])  # This adds a subplot that spans the first row entirely
# ax4.set_title(r'Outdoor temperature', fontsize=12)
# ax4.plot(outdoor_cool, label=r'Outdoor temperature', color='black', linewidth=2)
# ax4.legend(loc="lower right")
# ax4.set_ylabel('Temperature (°C)', fontsize=12)
# ax4.set_xticklabels([])
###################################################
###thermal weight
ax5 = fig.add_subplot(gs[1, 0:2])  # This adds a subplot that spans the first row entirely
ax5.set_title(r'$\delta=0$ VS $\delta=17.5$ VS $\delta=35$', fontsize=17)

ax5.plot(weight_heat_benchmark_delta_0, label=r'$\delta=0$', color='green', linewidth=2)
ax5.plot(weight_heat, label=r'$\delta=17.5$', color='cyan', linewidth=2)
ax5.plot(weight_heat_benchmark_delta_35, label=r'$\delta=35$', color='blue', linewidth=2)
specific_dates = ['Jan 17', 'Jan 19', 'Jan 21', 'Jan 23', 'Jan 25', 'Jan 27', 'Jan 29', 'Jan 31']
index_dates = [i * 96 * 2 for i in range(len(specific_dates))]  # Each day has 96 steps (4 steps per hour * 24 hours)
ax5.set_xticks(index_dates)
ax5.legend(loc="lower left", fontsize=16)
ax5.set_xticklabels(specific_dates,fontsize=12)
ax5.set_ylabel(r'$\alpha(t)$', fontsize=20)

ax5_twin = ax5.twinx()
ax5_twin.plot(outdoor_heat, 'black', linestyle='--', label='Outdoor Temperature')
ax5_twin.set_ylabel('Outdoor Temperature (°C)', fontsize=16, color='black')
ax5_twin.legend(loc="upper right", fontsize=16)



ax6 = fig.add_subplot(gs[1, 2:4])  # This adds a subplot that spans the first row entirely
ax6.set_title(r'$\delta=0$ VS $\delta=17.5$ VS $\delta=35$', fontsize=17)

ax6.plot(weight_cool_benchmark_delta_0, label=r'$\alpha(t)-\delta=0$', color='green', linewidth=2)
ax6.plot(weight_cool, label=r'$\alpha(t)-\delta=17.5$', color='cyan', linewidth=2)
ax6.plot(weight_cool_benchmark_delta_35, label=r'$\alpha(t)-\delta=35$', color='blue', linewidth=2)
ax6.legend(loc="lower left", fontsize=16)
specific_dates2 = ['Oct 09', 'Oct 11', 'Oct 13', 'Oct 15', 'Oct 17', 'Oct 20', 'Oct 22', 'Oct 24']
index_dates2 =  [i * 24 * 2 for i in range(len(specific_dates2))]
ax6.set_xticks(index_dates2)
ax6.set_xticklabels(specific_dates2,fontsize=12)
ax6.set_ylabel(r'$\alpha$', fontsize=20)

ax6_twin = ax6.twinx()
ax6_twin.plot(outdoor_cool, 'black', linestyle='--', label='Outdoor Temperature')
ax6_twin.set_ylabel('Outdoor Temperature (°C)', fontsize=16, color='black')
ax6_twin.legend(loc="upper right", fontsize=16)

# 3rd row with three individual plots
##############################################################################################################
title = 'Thermal discomfort KPI'  # 'Energy usage KPI'
colors = ['green', 'cyan', 'blue']
labels = [r'$\delta=0$', r'$\delta=17.5$',r'$\delta=35$']

# Make sure 'alls' contains exactly three values to match the labels and colors
alls = [thermal_kpi_heat_benchmark_delta_0[-1], thermal_kpi_heat[-1], thermal_kpi_heat_benchmark_delta_35[-1]]

# Ensure the lengths match
if len(labels) != len(alls) or len(labels) != len(colors):
    raise ValueError("The lengths of 'labels', 'alls', and 'colors' must match")

# Create the bar plot
ax = fig.add_subplot(gs[2, 0])
rects = ax.bar(labels, alls, color=colors)

# Set the ylabel
ax.set_ylabel(title, fontsize=16)

# Adding value annotations
for rect, value in zip(rects, alls):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2., height, f'{value:.3f}',
            ha='center', va='bottom', fontsize=14)

# Adjust y-limits to accommodate text annotations
max_height = max(alls)
ax.set_ylim(0, max_height + 0.1 * max_height)

# Set x-axis label size
ax.tick_params(axis='x', labelsize=12)
##############################################################################################################
title = 'Energy usage KPI'  # 'Energy usage KPI'
colors = ['green', 'cyan',  'blue']
labels = [r'$\delta=0$', r'$\delta=17.5$', r'$\delta=35$']
alls_cool_energy = [energy_kpi_heat_benchmark_delta_0[-1],  energy_kpi_heat[-1], energy_kpi_heat_benchmark_delta_35[-1]]

# Create the subplot
ax_cool = fig.add_subplot(gs[2, 1])
rects_cool = ax_cool.bar(labels, alls_cool_energy, color=colors)
ax_cool.set_ylabel(title, fontsize=16)
# Add value annotations above the bars
for rect, value in zip(rects_cool, alls_cool_energy):
    height = rect.get_height()
    ax_cool.text(rect.get_x() + rect.get_width() / 2., height, f'{value:.3f}',
            ha='center', va='bottom', fontsize=14)

# Adjust y-limits to accommodate text annotations
max_height = max(alls_cool_energy)
ax_cool.set_ylim(0, max_height + 0.1 * max_height)

ax_cool.tick_params(axis='x', labelsize=12)
##############################################################################################################
title = 'Thermal discomfort KPI'  # 'Energy usage KPI'
colors = ['green', 'cyan', 'blue']
labels = [r'$\delta=0$', r'$\delta=17.5$', r'$\delta=35$']
alls_cool = [thermal_kpi_cool_benchmark_delta_35[-1], thermal_kpi_cool[-1],  thermal_kpi_cool_benchmark_delta_0[-1],]

# Create the subplot
ax_cool = fig.add_subplot(gs[2, 2])

# Plot the bars
rects_cool = ax_cool.bar(labels, alls_cool, color=colors)
# Set the ylabel
ax_cool.set_ylabel(title, fontsize=16)
# Add value annotations above the bars
for rect, value in zip(rects_cool, alls_cool):
    height = rect.get_height()
    ax_cool.text(rect.get_x() + rect.get_width() / 2., height, f'{value:.3f}',
            ha='center', va='bottom', fontsize=14)

# Adjust y-limits to accommodate text annotations
max_height = max(alls_cool)
ax_cool.set_ylim(0, max_height + 0.1 * max_height)

ax_cool.tick_params(axis='x', labelsize=12)

##############################################################################################################
title = 'Energy usage KPI'  # 'Energy usage KPI'
colors = ['green', 'cyan',  'blue']
labels = [r'$\delta=0$', r'$\delta=17.5$', r'$\delta=35$']
alls_cool_energy = [energy_kpi_cool_benchmark_delta_0[-1],  energy_kpi_cool[-1], energy_kpi_cool_benchmark_delta_35[-1]]

# Create the subplot
ax_cool = fig.add_subplot(gs[2, 3])
rects_cool = ax_cool.bar(labels, alls_cool_energy, color=colors)
ax_cool.set_ylabel(title, fontsize=16)
# Add value annotations above the bars
for rect, value in zip(rects_cool, alls_cool_energy):
    height = rect.get_height()
    ax_cool.text(rect.get_x() + rect.get_width() / 2., height, f'{value:.3f}',
            ha='center', va='bottom', fontsize=14)

# Adjust y-limits to accommodate text annotations
max_height = max(alls_cool_energy)
ax_cool.set_ylim(0, max_height + 0.1 * max_height)

ax_cool.tick_params(axis='x', labelsize=12)

plt.tight_layout()
plt.savefig('C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\varying\\delta_visual.png')


plt.show()
