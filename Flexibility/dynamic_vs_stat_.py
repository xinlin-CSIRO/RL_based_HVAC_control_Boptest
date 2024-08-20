import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




debug_model=1
occup=1
cool_or_heat='heat'#or 'heat'
file=r"C:\Users\wan397\OneDrive - CSIRO\Desktop\RL_WORK_SUMMARY\2025 Conference\all_results_in_one.xlsx"
data_source=pd.ExcelFile(file)


#our method
our_method = pd.read_excel(data_source, 'Peak_heat_our')
head = 14400
end = len(our_method)-1
indoor_our_approach=np.array(our_method['indoor_temperature'][head:end]).astype(float)
all_conumsption_our_approach=np.array(our_method['all_consumption'][head:end]).astype(float)
thermal_kpi_our_list=np.array(our_method['themal_kpi'][head:end]).astype(float)
thermal_kpi_our_approach=float(thermal_kpi_our_list[-1]) - float(thermal_kpi_our_list[0])
energy_kpi_our_list=np.array(our_method['energy_kpi'][head:end]).astype(float)
energy_kpi_our_approach=float(energy_kpi_our_list[-1]) - float(energy_kpi_our_list[0])
cost_kpi_our_list=np.array(our_method['cost_kpi'][head:end]).astype(float)
cost_kpi_our_approach=float(cost_kpi_our_list[-1]) - float(cost_kpi_our_list[0])
outdoor_temp=np.array(our_method['outdoor_air'][head:end]).astype(float)

#rule_based
rule_based = pd.read_excel(data_source, 'Peak_heat_pid')
head = 0
end = len(rule_based) - 1
indoor_benchmark_rule=np.array(rule_based['indoor_temperature'][head:end])
all_conumsption_benchmark_rule=np.array(rule_based['all_consumption'][head:end])
thermal_kpi_benchmark_rule=np.array(rule_based['themal_kpi'][head:end])
price=np.array(rule_based['price'][head:end])
up_boundary=np.array(rule_based['boundary_1'][head:end])
low_boundary=np.array(rule_based['boundary_0'][head:end])

thermal_kpi_rule_list=np.array(rule_based['themal_kpi'][head:end]).astype(float)
thermal_kpi_rule=float(thermal_kpi_rule_list[-1])
energy_kpi_rule_list=np.array(rule_based['energy_kpi'][head:end]).astype(float)
energy_kpi_rule=float(energy_kpi_rule_list[-1])
cost_kpi_rule_list=np.array(rule_based['cost_kpi'][head:end]).astype(float)
cost_kpi_rule=float(cost_kpi_rule_list[-1])

rl_1 = pd.read_excel(data_source, 'Peak_heat_our_fixed')
head = 14400
end = len(rl_1) - 1
indoor_rl_1 = np.array(rl_1['indoor_temperature'][head:end]).astype(float)
all_conumsption_rl_1 = np.array(rl_1['all_consumption'][head:end]).astype(float)
thermal_kpi_rl_1 = np.array(rl_1['themal_kpi'][head:end]).astype(float)
thermal_kpi_rl_1 = float(thermal_kpi_rl_1[-1]) - float(thermal_kpi_rl_1[0])
energy_kpi_rl_1 = np.array(rl_1['energy_kpi'][head:end]).astype(float)
energy_kpi_rl_1 = float(energy_kpi_rl_1[-1]) - float(energy_kpi_rl_1[0])
cost_kpi_rl_1 = np.array(rl_1['cost_kpi'][head:end]).astype(float)
cost_kpi_rl_1 = float(cost_kpi_rl_1[-1]) - float(cost_kpi_rl_1[0])

print('done')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Create a figure with GridSpec layout
fig = plt.figure(figsize=(33, 19))
gs = gridspec.GridSpec(3, 2, figure=fig)
# fig.suptitle(r'Dynamic thermal weight $\alpha(t)$ V.S. fixed thermal weight $\alpha$=0.5',y=0.915)  # Set a title for the whole figure

ax1 = fig.add_subplot(gs[0, 0:2])  # This adds a subplot that spans the first row entirely
# ax1.set_title('Best Hydronic Heat Pump: Peak heat days', fontsize=17)
ax1.set_title('BOPTEST, Bestest air, Peak heat days scenario', fontsize=24)
ax1.plot(indoor_our_approach, label=r'Indoor temperature - $\alpha(t)$', color='cyan', linewidth=2)
ax1.plot(indoor_rl_1, label=r'Indoor temperature - $\alpha=0.5$', color='blue', linewidth=2)
ax1.plot(up_boundary, 'g:')
ax1.plot(low_boundary, 'g:')
ax1.legend(loc="lower right", fontsize=16)
# specific_dates = ['Jan 17', 'Jan 19', 'Jan 21', 'Jan 23', 'Jan 25', 'Jan 27', 'Jan 29', 'Jan 31']
# index_dates = [i * 96 * 2 for i in range(len(specific_dates))]  # Each day has 96 steps (4 steps per hour * 24 hours)
# ax1.set_xticks(index_dates)
ax1.set_xticklabels([])
ax1.set_ylabel('Temperature (°C)', fontsize=16)

ax2 = fig.add_subplot(gs[0:1, 2:4])  # This adds a subplot that spans the first row entirely
ax2.set_title('Test Case 2 - Cooling dominated scenario', fontsize=24)
ax2.plot(best_air_indoor_alpt_t, label=r'Indoor temperature - $\alpha(t)$', color='cyan', linewidth=2)
ax2.plot(best_air_indoor_alpt_fixed, label=r'Indoor temperature - $\alpha=0.5$', color='blue', linewidth=2)
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
ax5.set_title(r'$\alpha(t)$ VS $\alpha=0.5$', fontsize=17)
ax5.plot(weight_heat, label=r'$\alpha(t)$', color='cyan', linewidth=2)
ax5.plot(weight_heat_benchmark, label=r'$\alpha=0.5$', color='blue', linewidth=2)
specific_dates = ['Jan 17', 'Jan 19', 'Jan 21', 'Jan 23', 'Jan 25', 'Jan 27', 'Jan 29', 'Jan 31']
index_dates = [i * 96 * 2 for i in range(len(specific_dates))]  # Each day has 96 steps (4 steps per hour * 24 hours)
ax5.set_xticks(index_dates)
ax5.legend(loc="lower left", fontsize=16)
ax5.set_xticklabels(specific_dates,fontsize=12)
ax5.set_ylabel(r'$\alpha$', fontsize=20)

ax5_twin = ax5.twinx()
ax5_twin.plot(outdoor_heat, 'black', linestyle='--', label='Outdoor Temperature')
ax5_twin.set_ylabel('Outdoor Temperature (°C)', fontsize=16, color='black')
ax5_twin.legend(loc="upper right", fontsize=16)



ax6 = fig.add_subplot(gs[1, 2:4])  # This adds a subplot that spans the first row entirely
ax6.set_title(r'$\alpha(t)$ VS $\alpha=0.5$', fontsize=17)
ax6.plot(weight_cool, label=r'$\alpha(t)$', color='cyan', linewidth=2)
ax6.plot(weight_cool_benchmark, label=r'$\alpha=0.5$', color='blue', linewidth=2)
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
titles = 'Thermal discomfort KPI' #'Energy usage KPI'
colors = ['cyan','blue' ]
labels = [r'our method-$\alpha(t)$', r'our method-$\alpha=0.5$']
datasets = [thermal_kpi_heat[-1], thermal_kpi_cool[-1]]
# benchmarks = [[thermal_kpi_alpa_5_best_heat ],  #thermal_kpi_alpa_1_best_heat
#               [thermal_kpi_alpa_5_best_air ]]
benchmarks = [[thermal_kpi_heat_benchmark[-1] ],  #thermal_kpi_alpa_1_best_heat
              [thermal_kpi_cool_benchmark[-1] ]]
loc=np.array([0,2])
x=0
for i in loc:
    ax = fig.add_subplot(gs[2, i])
    # ax.set_title(titles[i])
    all_values = [datasets[x]] + benchmarks[x]
    rects = ax.bar(labels, all_values, color=colors, linewidth=2)
    # ax.legend(loc="lower right")
    ax.set_ylabel(titles,fontsize=16)
    x+=1

    # Adding value annotations
    for rect, value in zip(rects, all_values):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., height, f'{value:.2f}',
                ha='center', va='bottom', fontsize=14)

    # Adjust y-limits to accommodate text annotations
    max_height = max(all_values)
    ax.set_ylim(0, max_height + 0.1 * max_height)

    ax.tick_params(axis='x', labelsize=12)


# Third row with three individual plots
titles = 'Energy usage KPI'
colors = ['cyan','blue' ]
labels = [r'our method-$\alpha(t)$', r'our method-$\alpha=0.5$']
datasets = [energy_kpi_heat[-1], energy_kpi_cool[-1]]
benchmarks = [[energy_kpi_heat_benchmark[-1] ],  #thermal_kpi_alpa_1_best_heat
              [energy_kpi_cool_benchmark[-1] ]]
loc2=np.array([1,3])
y=0
for i in loc2:
    ax = fig.add_subplot(gs[2, i])
    # ax.set_title(titles[i])
    all_values = [datasets[y]] + benchmarks[y]
    rects = ax.bar(labels, all_values, color=colors, linewidth=2)
    # ax.legend(loc="lower right")
    ax.set_ylabel(titles,fontsize=16)
    y+=1

    # Adding value annotations
    for rect, value in zip(rects, all_values):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., height, f'{value:.2f}',
                ha='center', va='bottom', fontsize=14)

    # Adjust y-limits to accommodate text annotations
    max_height = max(all_values)
    ax.set_ylim(0, max_height + 0.1 * max_height)

    ax.tick_params(axis='x', labelsize=12)

plt.tight_layout()
plt.savefig('C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\varying\\alpa.png')
