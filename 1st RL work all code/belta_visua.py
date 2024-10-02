import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Function to calculate alpha(t) based on the formula
def calculate_alpha(t_out, t_ref, beta, delta):
    return 1 / (1 + np.exp(beta * (np.abs(t_out - t_ref) - delta)))

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


# Parameters
out_temp = np.linspace(-30, 30, 500)
out_temp_summer = np.linspace(-30, 30, 500)
beta_winter = -1
beta_summer = 1
ref_day = 22.5 * np.ones(500)
ref_winter_night = 15 * np.ones(500)
ref_night_summer = 30 * np.ones(500)
delta_values = [0, 17.5, 35]
colors = ['green', 'cyan', 'blue']  # Colors for different delta values


def alpha(outdoor_temp, ref, beta, delta):
    return 1 / (1 + np.exp(beta * (abs(outdoor_temp-ref) - delta)))



# fig.suptitle(r'Dynamic thermal weight $\alpha(t)$ V.S. fixed thermal weight $\alpha$=0.5',y=0.915)  # Set a title for the whole figure


fig = plt.figure(figsize=(30, 23))
gs = gridspec.GridSpec(4, 4, figure=fig)

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
ax0 = fig.add_subplot(gs[0:1, 0:2])
for delta, color in zip(delta_values, colors):
    alpha_winter_day = alpha(out_temp, ref_day, beta_winter, delta)
    alpha_winter_night = alpha(out_temp, ref_winter_night, beta_winter, delta)
    ax0.plot(x_values_winter_day, alpha_winter_day,label=r'$\alpha(t)$ for winter $(\delta = {}^\circ \mathrm{{C}})$'.format(delta),  color=color, linestyle='-', linewidth=4)


# Titles and labels
ax0.set_title(r'Thermal weight $\alpha(t)$ for different $\delta$ values in winter seasons', fontsize=18)
ax0.set_xlabel(r'The absolute difference $|t_{out} - t_{ref}|$', fontsize=14)
ax0.set_ylabel(r'$\alpha(t)$', fontsize=20)
ax0.legend(fontsize=12)
ax0.grid(True)

# Plot for summer season
ax01 = fig.add_subplot(gs[0:1, 2:4])
for delta, color in zip(delta_values, colors):
    alpha_summer_day = alpha(out_temp_summer, ref_day, beta_summer, delta)
    alpha_summer_night = alpha(out_temp_summer, ref_night_summer, beta_summer, delta)
    ax01.plot(x_values_summer_day, alpha_summer_day, label=r'$\alpha(t)$ for summer $(\delta = {}^\circ \mathrm{{C}})$'.format(delta),  color=color, linestyle='-', linewidth=4)
    # ax01.plot(x_values_summer_night, alpha_summer_night, label=r'$\alpha(t)$ for summer $(\delta = {}$)  - unoccupied hours'.format(delta), color=color, linestyle='--', linewidth=4)

# Titles and labels
ax01.set_title(r'Thermal weight $\alpha(t)$ for different $\delta$ values in summer seasons', fontsize=18)
ax01.set_xlabel(r'The absolute difference $|t_{out} - t_{ref}|$', fontsize=14)
ax01.set_ylabel(r'$\alpha(t)$', fontsize=20)
ax01.legend(fontsize=12)
ax01.grid(True)



# Plot for Test Case 1 - Heating scenario
ax1 = fig.add_subplot(gs[1:2, 0:2])
ax1.set_title('Test Case 1 - Heating scenario', fontsize=18)
ax1.plot(best_heat_indoor_delta_0, label=r'Indoor temperature - $\delta=0^{\circ}\mathrm{C} $', color='green', linewidth=2)
ax1.plot(best_heat_indoor_alpt_t, label=r'Indoor temperature - $\delta=17.5^{\circ}\mathrm{C} $', color='cyan', linewidth=2)
ax1.plot(best_heat_indoor_delta_35, label=r'Indoor temperature - $\delta=35^{\circ}\mathrm{C} $', color='blue', linewidth=2)
ax1.plot(up_boundary_heat, 'g:')
ax1.plot(low_boundary_heat, 'g:')
ax1.legend(loc="lower right", fontsize=12)
ax1.set_xticklabels([])
ax1.set_ylabel('Temperature (°C)', fontsize=14)

# Plot for Test Case 2 - Cooling dominated scenario
ax2 = fig.add_subplot(gs[1:2, 2:4])
ax2.set_title('Test Case 2 - Cooling dominated scenario', fontsize=18)
ax2.plot(best_cool_indoor_delta_0, label=r'Indoor temperature - $\delta=0^{\circ}\mathrm{C}$', color='green', linewidth=2)
ax2.plot(best_cool_indoor_alpt_t, label=r'Indoor temperature - $\delta=17.5^{\circ}\mathrm{C}$', color='cyan', linewidth=2)
ax2.plot(best_cool_indoor_delta_35, label=r'Indoor temperature - $\delta=35^{\circ}\mathrm{C}$', color='blue', linewidth=2)
ax2.plot(up_boundary_cool, 'g:')
ax2.plot(low_boundary_cool, 'g:')
ax2.legend(loc="lower right", fontsize=12)
ax2.set_xticklabels([])
ax2.set_ylabel('Temperature (°C)', fontsize=14)

# Plot thermal weight for Test Case 1
ax5 = fig.add_subplot(gs[2, 0:2])
ax5.set_title(r'$\delta=0$ VS $\delta=17.5$ VS $\delta=35$', fontsize=14)
ax5.plot(weight_heat_benchmark_delta_0, label=r'$\delta=0$', color='green', linewidth=2)
ax5.plot(weight_heat, label=r'$\delta=17.5$', color='cyan', linewidth=2)
ax5.plot(weight_heat_benchmark_delta_35, label=r'$\delta=35$', color='blue', linewidth=2)
specific_dates = ['Jan 17', 'Jan 19', 'Jan 21', 'Jan 23', 'Jan 25', 'Jan 27', 'Jan 29', 'Jan 31']
index_dates = [i * 96 * 2 for i in range(len(specific_dates))]
ax5.set_xticks(index_dates)
ax5.legend(loc="lower left", fontsize=12)
ax5.set_xticklabels(specific_dates, fontsize=10)
ax5.set_ylabel(r'$\alpha(t)$', fontsize=14)
ax5_twin = ax5.twinx()
ax5_twin.plot(outdoor_heat, 'black', linestyle='--', label='Outdoor Temperature')
ax5_twin.set_ylabel('Outdoor Temperature (°C)', fontsize=14, color='black')
ax5_twin.legend(loc="upper right", fontsize=12)

# Plot thermal weight for Test Case 2
ax6 = fig.add_subplot(gs[2, 2:4])
ax6.set_title(r'$\delta=0^{\circ}\mathrm{C}$ VS $\delta=17.5^{\circ}\mathrm{C}$ VS $\delta=35^{\circ}\mathrm{C}$', fontsize=14)
ax6.plot(weight_cool_benchmark_delta_0, label=r'$\alpha(t)-\delta=0^{\circ}\mathrm{C}$', color='green', linewidth=2)
ax6.plot(weight_cool, label=r'$\alpha(t)-\delta=17.5^{\circ}\mathrm{C}$', color='cyan', linewidth=2)
ax6.plot(weight_cool_benchmark_delta_35, label=r'$\alpha(t)-\delta=35^{\circ}\mathrm{C}$', color='blue', linewidth=2)
ax6.legend(loc="lower left", fontsize=12)
specific_dates2 = ['Oct 09', 'Oct 11', 'Oct 13', 'Oct 15', 'Oct 17', 'Oct 20', 'Oct 22', 'Oct 24']
index_dates2 = [i * 24 * 2 for i in range(len(specific_dates2))]
ax6.set_xticks(index_dates2)
ax6.set_xticklabels(specific_dates2, fontsize=10)
ax6.set_ylabel(r'$\alpha(t)$', fontsize=14)
ax6_twin = ax6.twinx()
ax6_twin.plot(outdoor_cool, 'black', linestyle='--', label='Outdoor Temperature')
ax6_twin.set_ylabel('Outdoor Temperature (°C)', fontsize=14, color='black')
ax6_twin.legend(loc="upper right", fontsize=12)

# Plot thermal discomfort KPI for heating scenario
title = 'Thermal discomfort KPI'
colors = ['green', 'cyan', 'blue']
labels = [r'$\delta=0^{\circ}\mathrm{C}$', r'$\delta=17.5^{\circ}\mathrm{C}$', r'$\delta=35^{\circ}\mathrm{C}$']
alls = [thermal_kpi_heat_benchmark_delta_0[-1], thermal_kpi_heat[-1], thermal_kpi_heat_benchmark_delta_35[-1]]
ax = fig.add_subplot(gs[3, 0])
rects = ax.bar(labels, alls, color=colors)
ax.set_ylabel(title, fontsize=14)
for rect, value in zip(rects, alls):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2., height, f'{value:.2f}', ha='center', va='bottom', fontsize=12)
max_height = max(alls)
ax.set_ylim(0, max_height + 0.1 * max_height)
ax.tick_params(axis='x', labelsize=12)

# Plot energy usage KPI for heating scenario
title = 'Energy usage KPI'
alls_cool_energy = [energy_kpi_heat_benchmark_delta_0[-1], energy_kpi_heat[-1], energy_kpi_heat_benchmark_delta_35[-1]]
ax_cool = fig.add_subplot(gs[3, 1])
rects_cool = ax_cool.bar(labels, alls_cool_energy, color=colors)
ax_cool.set_ylabel(title, fontsize=14)
for rect, value in zip(rects_cool, alls_cool_energy):
    height = rect.get_height()
    ax_cool.text(rect.get_x() + rect.get_width() / 2., height, f'{value:.2f}', ha='center', va='bottom', fontsize=12)
max_height = max(alls_cool_energy)
ax_cool.set_ylim(0, max_height + 0.1 * max_height)
ax_cool.tick_params(axis='x', labelsize=12)

# Plot thermal discomfort KPI for cooling scenario
title = 'Thermal discomfort KPI'
alls_cool = [thermal_kpi_cool_benchmark_delta_35[-1], thermal_kpi_cool[-1], thermal_kpi_cool_benchmark_delta_0[-1]]
ax_cool = fig.add_subplot(gs[3, 2])
rects_cool = ax_cool.bar(labels, alls_cool, color=colors)
ax_cool.set_ylabel(title, fontsize=14)
for rect, value in zip(rects_cool, alls_cool):
    height = rect.get_height()
    ax_cool.text(rect.get_x() + rect.get_width() / 2., height, f'{value:.2f}', ha='center', va='bottom', fontsize=12)
max_height = max(alls_cool)
ax_cool.set_ylim(0, max_height + 0.1 * max_height)
ax_cool.tick_params(axis='x', labelsize=12)

# Plot energy usage KPI for cooling scenario
title = 'Energy usage KPI'
alls_cool_energy = [energy_kpi_cool_benchmark_delta_0[-1], energy_kpi_cool[-1], energy_kpi_cool_benchmark_delta_35[-1]]
ax_cool = fig.add_subplot(gs[3, 3])
rects_cool = ax_cool.bar(labels, alls_cool_energy, color=colors)
ax_cool.set_ylabel(title, fontsize=14)
for rect, value in zip(rects_cool, alls_cool_energy):
    height = rect.get_height()
    ax_cool.text(rect.get_x() + rect.get_width() / 2., height, f'{value:.2f}', ha='center', va='bottom', fontsize=12)
max_height = max(alls_cool_energy)
ax_cool.set_ylim(0, max_height + 0.1 * max_height)
ax_cool.tick_params(axis='x', labelsize=12)

plt.tight_layout()
plt.savefig('C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\varying\\delta_visual.png')
plt.show()
