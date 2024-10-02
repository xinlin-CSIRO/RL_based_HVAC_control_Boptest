import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Assuming the file path and file reading parts remain unchanged
file = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\varying\\"
our_approach = file + 'final_results_and_code.xlsx'
data_source=pd.ExcelFile(our_approach)
our_method = pd.read_excel(data_source, 'best_heat_alpa(t)')
head=9600
end=len(our_method)-1
indoor_our_approach=np.array(our_method['indoor_temp'][head:end]).astype(float)

up_boundary = np.array(our_method['boundary_1'][head:end]).astype(float)
low_boundary = np.array(our_method['boundary_0'][head:end]).astype(float)
outdoor = np.array(our_method['outdoor_air'][head:end]).astype(float)
actions = np.array(our_method['action'][head:end]).astype(float)
actions_series = pd.Series(actions)
window_size = 10  # This is an example window size; adjust based on your data specifics
action_ =actions_series.rolling(window=window_size).mean()
step_rewards=np.array(our_method['reward'][head:end]).astype(float)


thermal_kpi_our_approach = np.array(our_method['themal_kpi'][head:end]).astype(float)

energy_kpi_our_approach = np.array(our_method['energy_kpi'][head:end]).astype(float)
font_size=20

fig = plt.figure(figsize=(26, 10))
gs = gridspec.GridSpec(2, 2, figure=fig)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(indoor_our_approach, 'cyan', label='Measured indoor temperature', linewidth=3)
ax1.plot(action_, 'blue', linestyle=':', label='Actions: Zone operative temperature setpoint', alpha=0.7)
ax1.plot(up_boundary, 'silver', label='Upper boundary')
ax1.plot(low_boundary, 'silver', label='Lower boundary')
ax1.set_ylabel('Temperature (°C)', fontsize=font_size)
ax1.legend(loc="lower right", fontsize=font_size)
ax1.set_xticklabels([])



ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(outdoor, 'black', label='Outdoor air temperature')
ax2.set_ylabel('Temperature (°C)', fontsize=font_size)
ax2.legend(loc="lower right", fontsize=font_size)
specific_dates = ['Jan 17', 'Jan 19', 'Jan 21', 'Jan 23', 'Jan 25', 'Jan 27', 'Jan 29', 'Jan 31']
ax2.set_xticklabels(specific_dates)




# Assuming the file path and file reading parts remain unchanged
our_method2 = pd.read_excel(data_source, 'best_air_apla(t)')
head=2400
end=len(our_method2)-1
indoor_our_approach2=np.array(our_method2['indoor_temp'][head:end]).astype(float)

up_boundary = np.array(our_method2['boundary_1'][head:end]).astype(float)
low_boundary = np.array(our_method2['boundary_0'][head:end]).astype(float)
outdoor = np.array(our_method2['outdoor_air'][head:end]).astype(float)
actions = np.array(our_method2['action_0'][head:end]).astype(float)
thermal_kpi_our_approach = np.array(our_method2['themal_kpi'][head:end]).astype(float)
energy_kpi_our_approach = np.array(our_method2['energy_kpi'][head:end]).astype(float)
# step_rewards_2=np.array(our_method2['reward'][0:2300]).astype(float)
file2 = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\2025 Conference\\all_results_in_one.xlsx"
#
our_method2_w = pd.read_excel(file2, 'Peak_cool_our')
step_rewards_2=np.array(our_method2_w['reward'][0:9500]).astype(float)+ 1*np.ones(9500)


ax11 = fig.add_subplot(gs[0, 1])
ax11.plot(indoor_our_approach2, 'cyan', label='Measured indoor temperature', linewidth=3)
ax11.plot(actions, 'blue', linestyle=':', label='Actions: Zone operative temperature setpoint', alpha=0.7)
ax11.plot(up_boundary, 'silver', label='Upper boundary')
ax11.plot(low_boundary, 'silver', label='Lower boundary')
ax11.set_ylabel('Temperature (°C)', fontsize=font_size)
ax11.legend(loc="lower right", fontsize=font_size)
ax11.set_xticklabels([])



ax22 = fig.add_subplot(gs[1, 1])
ax22.plot(outdoor, 'black', label='Outdoor air temperature')
ax22.set_ylabel('Temperature (°C)', fontsize=font_size)
ax22.legend(loc="lower right", fontsize=font_size)
specific_dates2 = ['Oct 09', 'Oct 11', 'Oct 13', 'Oct 15', 'Oct 17', 'Oct 20', 'Oct 22', 'Oct 24']
ax22.set_xticklabels(specific_dates2)
plt.tight_layout()
plt.savefig('C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\varying\\results_from_boptest.png')
plt.show()
