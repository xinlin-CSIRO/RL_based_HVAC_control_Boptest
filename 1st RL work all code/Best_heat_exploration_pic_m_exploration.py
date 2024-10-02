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


plt.plot(boundary_0_best_heat, 'g:', label='boundary')
plt.plot(boundary_1_best_heat, 'g:', label='boundary')
plt.show()


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