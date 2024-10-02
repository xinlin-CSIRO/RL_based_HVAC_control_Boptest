import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file=r"C:\Users\wan397\OneDrive - CSIRO\Desktop\RL_WORK_SUMMARY\2025 Conference\CSIRO AI GYM SUMMER.xlsx"
data_source=pd.ExcelFile(file)

our_method = pd.read_excel(data_source, 'test_data')

end = len(our_method)-1
interval=10
head = int(end-(24*60/interval)*14)
indoor_our_approach=np.array(our_method['indoor_temperature'][head:end]).astype(float)
up_boundary=np.array(our_method['boundary_1'][head:end])
low_boundary=np.array(our_method['boundary_0'][head:end])
outdoor_temp=np.array(our_method['outdoor_air'][head:end]).astype(float)
all_conumsption_our_approach=np.array(our_method['cooling_usage'][head:end]).astype(float)
thermal_kpi=np.array(our_method['thermal_discomfort_kpi'][head:end]).astype(float)
thermal_kpi=thermal_kpi-thermal_kpi[0]
cooling_kpi=np.array(our_method['cooling_kpi'][head:end]).astype(float)
cooling_kpi=cooling_kpi-cooling_kpi[0]
cooling_cost_kpi=np.array(our_method['cooling_cost_kpi'][head:end]).astype(float)
cooling_cost_kpi=cooling_cost_kpi-cooling_cost_kpi[0]
price=np.array(our_method['price'][head:end]).astype(float)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(5, 1, figure=fig)


ax1 = fig.add_subplot(gs[0:2, :])  # This adds a subplot that spans the first row entirely
ax1.set_title('CSIRO AI Gym: Cooling scenario')
ax1.plot(indoor_our_approach, label='Indoor temperature of our method', color='plum', linewidth=2)
ax1.plot(outdoor_temp, color='black', label='Outdoor temperature', linewidth=2)
ax1.plot(up_boundary, ':',color='gray', label='Thermal boundaries')
ax1.plot(low_boundary, ':',color='gray')
ax1.legend(loc="upper right")
ax1.set_ylabel('Temperature (°C)')
ax1.set_xticklabels([])  # Disabling x-axis labels for ax1



ax2 = fig.add_subplot(gs[2, :])  # This adds a subplot that spans the second row entirely
ax2.set_title('Real-time cumulative thermal discomfort KPI')
ax2.plot(thermal_kpi, color='black', label='Thermal discomfort KPI')
ax2.set_ylabel('Thermal discomfort KPI')
ax2.legend(loc="lower right")
ax2.set_xticklabels([])  # Disabling x-axis labels for ax1

ax3 = fig.add_subplot(gs[3, :])  # This adds a subplot that spans the second row entirely
ax3.set_title('Real-time cumulative cooling consumption KPI')
ax3.plot(cooling_cost_kpi, label='Cooling energy usage KPI', color='black', linewidth=2)
ax3.legend(loc="lower right")
ax3.set_ylabel('Cooling energy usage KPI')
ax3.set_xticklabels([])  # Disabling x-axis labels for ax1

ax4 = fig.add_subplot(gs[4, :])  # This adds a subplot that spans the second row entirely
ax4.set_title('Real-time cumulative operational cost KPI')
ax4.plot(cooling_cost_kpi, label='Operational cost KPI', color='black', linewidth=2)
ax4.legend(loc="lower right")
ax4.set_ylabel('Operational cost KPI')

specific_dates = ['Dec 24', 'Dec 26', 'Dec 28', 'Dec 30', 'Jan 1', 'Jan 3', 'Jan 5', 'Jan 7']
index_dates = [i * (24*6) * 2 for i in range(len(specific_dates))]
ax4.set_xticks(index_dates)
ax4.set_xticklabels(specific_dates)

fig.tight_layout()
plt.savefig('C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\2025 Conference\\CSIRO_AI_GYM.png')
plt.show()
