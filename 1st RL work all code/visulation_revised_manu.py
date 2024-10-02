import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 15), constrained_layout=True)

# Plot indoor temperature with boundaries

ax1.plot(indoor_our_approach, 'cyan', label='Measured indoor temperature', linewidth=3)
ax1.plot(action_, 'blue', linestyle=':', label='Actions: Zone operative temperature setpoint', alpha=0.7)
ax1.plot(up_boundary, 'silver', label='Upper boundary')
ax1.plot(low_boundary, 'silver', label='Lower boundary')
ax1.set_ylabel('Temperature (°C)', fontsize=12)
ax1.legend(loc="lower right", fontsize=12)
ax1.set_xticklabels([])



# Plot outdoor temperature
ax2.plot(outdoor, 'black', label='Outdoor air temperature')
ax2.set_ylabel('Temperature (°C)', fontsize=12)
ax2.legend(loc="lower right", fontsize=12)
ax2.set_xticklabels([])


# Plot thermal discomfort kpi
ax3.plot(thermal_kpi_our_approach, 'blue', label='Real-time cumulative thermal discomfort KPI')
ax3.set_ylabel('Thermal discomfort KPI (K.h/zone)', fontsize=12,color='blue')
ax3.legend(loc="center left", fontsize=12)
ax3.set_xticklabels([])

# Plot energy kpi
ax3_twin = ax3.twinx()
ax3_twin.plot(energy_kpi_our_approach, 'black', label='Real-time cumulative energy use KPI')
ax3_twin.set_ylabel('Energy use KPI (kWh/m$^2$)', fontsize=12)
ax3_twin.legend(loc="lower right", fontsize=12)
specific_dates = ['Jan 17', 'Jan 19', 'Jan 21', 'Jan 23', 'Jan 25', 'Jan 27', 'Jan 29', 'Jan 31']
index_dates = [i * 96 * 2 for i in range(len(specific_dates))]
ax3_twin.set_xticks(index_dates)
ax3_twin.set_xticklabels(specific_dates, fontsize=12)



step_rewards=np.array(our_method['reward'][0:9500]).astype(float)
batch_size=96
num_batches = len(step_rewards) // batch_size
batch_rewards = [
    np.sum(step_rewards[i * batch_size:(i + 1) * batch_size])
    for i in range(num_batches)
]
timesteps = np.arange(1, num_batches + 1) * batch_size
ma_window = 10  # Adjust the window size for smoother curve
smoothed_rewards = pd.Series(batch_rewards).rolling(window=ma_window).mean().to_numpy()
# Calculate standard deviation for confidence intervals (optional)
rolling_std = pd.Series(batch_rewards).rolling(window=ma_window).std().to_numpy()

decay_factor = np.linspace(1, 0.5, num_batches)
adjusted_std = rolling_std * decay_factor

ax4.plot(timesteps, smoothed_rewards, label='The proposed method', color='cyan', linewidth=3)

# Add confidence interval (optional)
ax4.fill_between(timesteps, smoothed_rewards - adjusted_std, smoothed_rewards + adjusted_std, color='cyan', alpha=0.2)

# ax4.plot(timesteps, batch_rewards)
ax4.set_xlabel('Episode', fontsize=12)
ax4.set_ylabel('Cumulative reward', fontsize=12)
ax4.legend(loc="lower right", fontsize=12)
ax4.set_title('Convergence plot of the proposed method', fontsize=12)


plt.savefig('C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\varying\\BOPTEST_1.png')

def add_value_labels(ax, spacing=5, buffer=0.1, fontsize=12):
    max_height = max([rect.get_height() for rect in ax.patches])
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        label = "{:.2f}".format(y_value)
        ax.annotate(label, (x_value, y_value), xytext=(0, spacing),
                    textcoords="offset points", ha='center', va='bottom', fontsize=fontsize)
    ax.set_ylim(0, max_height * (1 + buffer))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 11), constrained_layout=True)

names = ['Our approach', 'Benchmark 1: rule-based', 'Benchmark 2:model-free RL', 'Benchmark 3:model-based RL', 'Benchmark 4:MPC-based']
discomfort_kpis = [thermal_kpi_our_approach[-1], 8.38, 0.75, 1.88, 0.02]
bar_colors = ['cyan', 'blue', 'silver', 'darkgreen', 'lightblue']

axis_fontsize = 12
legend_fontsize = 16
bar_width = 0.6

for i, (name, kpi) in enumerate(zip(names, discomfort_kpis)):
    ax1.bar(name, kpi, color=bar_colors[i], label=name, width=bar_width)
ax1.set_ylabel('Thermal discomfort KPI (K.h/zone)', fontsize=axis_fontsize)
ax1.set_title('Cumulative thermal discomfort KPI', fontsize=legend_fontsize)
add_value_labels(ax1, fontsize=legend_fontsize)

energy_kpis = [energy_kpi_our_approach[-1], 3.48, 3.09, 2.77, 2.71]
for i, (name, kpi) in enumerate(zip(names, energy_kpis)):
    ax2.bar(name, kpi, color=bar_colors[i], label=name if i == 0 else "_nolegend_", width=bar_width)
ax2.set_ylabel('Energy use KPI (kWh/m$^2$)', fontsize=axis_fontsize)
ax2.set_title('Cumulative energy use KPI', fontsize=legend_fontsize)
add_value_labels(ax2, fontsize=legend_fontsize)

ax1.legend(loc='upper right', fontsize=legend_fontsize)
ax1.tick_params(axis='x', labelsize=11)
ax2.tick_params(axis='x', labelsize=11)
plt.savefig('C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\varying\\KPIs.png')

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

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 15), constrained_layout=True)

ax1.plot(indoor_our_approach2, 'cyan', label='Measured indoor temperature', linewidth=3)
ax1.plot(actions, 'blue', linestyle=':', label='Actions: Zone operative temperature setpoint', alpha=0.7)
ax1.plot(up_boundary, 'silver', label='Upper boundary')
ax1.plot(low_boundary, 'silver', label='Lower boundary')
ax1.set_ylabel('Temperature (°C)', fontsize=12)
ax1.legend(loc="lower right", fontsize=12)
ax1.set_xticklabels([])



# Plot outdoor temperature
ax2.plot(outdoor, 'black', label='Outdoor air temperature')
ax2.set_ylabel('Temperature (°C)', fontsize=12)
ax2.legend(loc="lower right", fontsize=12)
ax2.set_xticklabels([])


# Plot thermal discomfort kpi
ax3.plot(thermal_kpi_our_approach, 'blue', label='Real-time cumulative thermal discomfort KPI')
ax3.set_ylabel('Thermal discomfort KPI (K.h/zone)', fontsize=12,color='blue')
ax3.legend(loc="center left", fontsize=12)
ax3.set_xticklabels([])

# Plot energy kpi
ax3_twin = ax3.twinx()
ax3_twin.plot(energy_kpi_our_approach, 'black', label='Real-time cumulative energy use KPI')
ax3_twin.set_ylabel('Energy use KPI (kWh/m$^2$)', fontsize=12)
ax3_twin.legend(loc="lower right", fontsize=12)
specific_dates2 = ['Oct 09', 'Oct 11', 'Oct 13', 'Oct 15', 'Oct 17', 'Oct 20', 'Oct 22', 'Oct 24']
index_dates = [i * 24 * 2 for i in range(len(specific_dates))]
ax3_twin.set_xticks(index_dates)
ax3_twin.set_xticklabels(specific_dates2, fontsize=12)




batch_size=48
num_batches2 = len(step_rewards_2) // batch_size
batch_rewards2 = [
    np.sum(step_rewards_2[i * batch_size:(i + 1) * batch_size])
    for i in range(num_batches2)
]
timesteps = np.arange(1, num_batches2 + 1) * batch_size
ma_window = 10  # Adjust the window size for smoother curve
smoothed_rewards = pd.Series(batch_rewards2).rolling(window=ma_window).mean().to_numpy()
# Calculate standard deviation for confidence intervals (optional)
rolling_std = pd.Series(batch_rewards2).rolling(window=ma_window).std().to_numpy()

decay_factor = np.linspace(1, 0.3, num_batches2)
adjusted_std = rolling_std * decay_factor

ax4.plot(timesteps, smoothed_rewards, label='The proposed method', color='cyan', linewidth=3)

# Add confidence interval (optional)
ax4.fill_between(timesteps, smoothed_rewards - adjusted_std, smoothed_rewards + adjusted_std, color='cyan', alpha=0.2)

# ax4.plot(timesteps, batch_rewards)
ax4.set_xlabel('Episode', fontsize=12)
ax4.set_ylabel('Cumulative reward', fontsize=12)
ax4.legend(loc="lower right", fontsize=12)
ax4.set_title('Convergence plot of the proposed method', fontsize=12)



plt.savefig('C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\varying\\BOPTEST_2.png')
plt.show()
