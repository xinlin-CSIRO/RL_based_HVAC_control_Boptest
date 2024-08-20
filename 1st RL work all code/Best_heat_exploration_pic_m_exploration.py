import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


debug_model=1
occup=1
cool_or_heat='heat'#or 'heat'
file=r"C:\Users\wan397\OneDrive - CSIRO\Desktop\RL_WORK_SUMMARY\varying\m_exploration.xlsx"
data_source=pd.ExcelFile(file)


our_method = pd.read_excel(data_source, 'best_air_m=0')
indoor_our_approach_best_heat=np.array(our_method['indoor_temp']).astype(float)
thermal_kpi_our_list=np.array(our_method['themal_kpi']).astype(float)
thermal_kpi_our_approach_best_heat=float(thermal_kpi_our_list[-1]) - float(thermal_kpi_our_list[0])
energy_kpi_our_list=np.array(our_method['energy_kpi']).astype(float)
energy_kpi_our_approach_best_heat=float(energy_kpi_our_list[-1]) - float(energy_kpi_our_list[0])

boundary_0_best_heat=np.array(our_method['boundary_0']).astype(float)
boundary_1_best_heat=np.array(our_method['boundary_1']).astype(float)


our_method2 = pd.read_excel(data_source, 'best_air_m=5')
indoor_our_approach_best_air=np.array(our_method2['indoor_temp']).astype(float)
thermal_kpi_our_list2=np.array(our_method2['themal_kpi']).astype(float)
thermal_kpi_our_approach_best_air=float(thermal_kpi_our_list2[-1]) - float(thermal_kpi_our_list2[0])
energy_kpi_our_list2=np.array(our_method2['energy_kpi']).astype(float)
energy_kpi_our_approach_best_air=float(energy_kpi_our_list2[-1]) - float(energy_kpi_our_list2[0])

boundary_0_best_air=np.array(our_method2['boundary_0']).astype(float)
boundary_1_best_air=np.array(our_method2['boundary_1']).astype(float)




print('done')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Create a figure with GridSpec layout
fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 4, figure=fig)
# fig.suptitle(r'Dynamic thermal weight $\alpha(t)$ V.S. fixed thermal weight $\alpha$=0.5',y=0.915)  # Set a title for the whole figure

plt.plot(indoor_our_approach_best_heat)
plt.plot(indoor_our_approach_best_air)

# ax1 = fig.add_subplot(gs[0:1, 0:4])  # This adds a subplot that spans the first row entirely
# # ax1.set_title('Best Hydronic Heat Pump: Peak heat days', fontsize=17)
# ax1.set_title('Test Case 1 - Heating scenario', fontsize=17)
# ax1.plot(indoor_our_approach_best_heat, label=r'Indoor temperature - $\alpha(t)$', color='cyan', linewidth=2)
# # ax1.plot(indoor_alpa_1_best_heat, label='Indoor temperature - $\alpha=0.1$', color='darkgreen', linewidth=2)
# ax1.plot(indoor_our_approach_best_air, label='Indoor temperature - $\alpha=0.5$', color='blue', linewidth=2)
plt.plot(boundary_0_best_heat, 'g:', label='boundary')
plt.plot(boundary_1_best_heat, 'g:', label='boundary')
plt.show()
# ax1.legend(loc="lower right")
# specific_dates = ['Jan 17', 'Jan 19', 'Jan 21', 'Jan 23', 'Jan 25', 'Jan 27', 'Jan 29', 'Jan 31']
# index_dates = [i * 96 * 2 for i in range(len(specific_dates))]  # Each day has 96 steps (4 steps per hour * 24 hours)
# ax1.set_xticks(index_dates)
# ax1.set_xticklabels(specific_dates,fontsize=12)
# ax1.set_ylabel('Temperature (°C)', fontsize=12)

# ax2 = fig.add_subplot(gs[0:1, 2:4])  # This adds a subplot that spans the first row entirely
# ax2.set_title('Test Case 2 - Cooling dominated scenario', fontsize=17)
# ax2.plot(indoor_our_approach_best_air, label=r'Indoor temperature - $\alpha(t)$', color='cyan', linewidth=2)
# # ax2.plot(indoor_alpa_1_best_air, label='Indoor temperature - $\alpha=0.1$', color='darkgreen', linewidth=2)
# ax2.plot(indoor_alpa_5_best_air, label='Indoor temperature - $\alpha=0.5$', color='blue', linewidth=2)
# ax2.plot(boundary_0_best_air, 'g:', label='boundary')
# ax2.plot(boundary_1_best_air, 'g:', label='boundary')
# # ax2.legend(loc="lower right")
# specific_dates2 = ['Oct 09', 'Oct 11', 'Oct 13', 'Oct 15', 'Oct 17', 'Oct 20', 'Oct 22', 'Oct 24']
# index_dates2 =  [i * 24 * 2 for i in range(len(specific_dates2))]
# ax2.set_xticks(index_dates2)
# ax2.set_xticklabels(specific_dates2,fontsize=12)
# ax2.set_ylabel('Temperature (°C)', fontsize=12)
#
# # 2nd row with three individual plots
# titles = 'Thermal discomfort KPI' #'Energy usage KPI'
# colors = ['cyan','blue' ]
# labels = [r'our method-$\alpha(t)$', r'our method-$\alpha=0.5$']
# datasets = [thermal_kpi_our_approach_best_heat, thermal_kpi_our_approach_best_air]
# benchmarks = [[thermal_kpi_alpa_5_best_heat ],  #thermal_kpi_alpa_1_best_heat
#               [thermal_kpi_alpa_5_best_air ]]
# loc=np.array([0,2])
# x=0
# for i in loc:
#     ax = fig.add_subplot(gs[1, i])
#     # ax.set_title(titles[i])
#     all_values = [datasets[x]] + benchmarks[x]
#     rects = ax.bar(labels, all_values, color=colors, linewidth=2)
#     # ax.legend(loc="lower right")
#     ax.set_ylabel(titles,fontsize=12)
#     x+=1
#
#     # Adding value annotations
#     for rect, value in zip(rects, all_values):
#         height = rect.get_height()
#         ax.text(rect.get_x() + rect.get_width() / 2., height, f'{value:.2f}',
#                 ha='center', va='bottom', fontsize=12)
#
#     # Adjust y-limits to accommodate text annotations
#     max_height = max(all_values)
#     ax.set_ylim(0, max_height + 0.1 * max_height)
#
#     ax.tick_params(axis='x', labelsize=11)
#
#
# # Third row with three individual plots
# titles = 'Energy usage KPI'
# colors = ['cyan','blue' ]
# labels = [r'our method-$\alpha(t)$', r'our method-$\alpha=0.5$']
# datasets = [energy_kpi_our_approach_best_heat, energy_kpi_our_approach_best_air]
# benchmarks = [[energy_kpi_alpa_5_best_heat ],  #thermal_kpi_alpa_1_best_heat
#               [energy_kpi_alpa_5_best_air ]]
# loc2=np.array([1,3])
# y=0
# for i in loc2:
#     ax = fig.add_subplot(gs[1, i])
#     # ax.set_title(titles[i])
#     all_values = [datasets[y]] + benchmarks[y]
#     rects = ax.bar(labels, all_values, color=colors, linewidth=2)
#     # ax.legend(loc="lower right")
#     ax.set_ylabel(titles,fontsize=12)
#     y+=1
#
#     # Adding value annotations
#     for rect, value in zip(rects, all_values):
#         height = rect.get_height()
#         ax.text(rect.get_x() + rect.get_width() / 2., height, f'{value:.2f}',
#                 ha='center', va='bottom', fontsize=12)
#
#     # Adjust y-limits to accommodate text annotations
#     max_height = max(all_values)
#     ax.set_ylim(0, max_height + 0.1 * max_height)
#
#     ax.tick_params(axis='x', labelsize=11)


plt.savefig('C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\varying\\m_exploration.png')


import numpy as np
import matplotlib.pyplot as plt

# Define the parameters
delta = 17.5
beta_winter = -1
beta_summer = 1

# Define the function for thermal weight alpha(t)
def alpha(t, beta, delta):
    return 1 / (1 + np.exp(beta * (t - delta)))

# Define the range for t
t = np.linspace(0, 35, 400)

# Calculate alpha(t) for winter and summer
alpha_winter = alpha(t, beta_winter, delta)
alpha_summer = alpha(t, beta_summer, delta)

# Create the plot
plt.figure(figsize=(9, 6))
plt.plot(t, alpha_winter, 'b', label=r'$\alpha(t)$ for winter seasons ($\beta=-1, \delta=17.5$)')
plt.plot(t, alpha_summer, 'g', label=r'$\alpha(t)$ for summer seasons ($\beta = 1, \delta=17.5$)')
plt.axvline(x=delta, color='k', linestyle='--')

# Set labels and title
plt.xlabel('The absolute difference between reference temperature and outdoor temperature: $|t_{{out}} - t_{{ref}}|$', fontsize=12)
plt.ylabel(r'Thermal weight $\alpha(t)$', fontsize=12)
plt.title(r'Thermal weight $\alpha(t)$ for winter and summer seasons', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)

plt.savefig('C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\varying\\Thermal_weight.png')


plt.show()
#