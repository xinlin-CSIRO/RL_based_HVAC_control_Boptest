import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# Define the parameters
delta = 17.5

beta_summer = 1

# Define the function for thermal weight alpha(t)
def alpha(outdoor_temp, ref, beta, delta):
    return 1 / (1 + np.exp(beta * (abs(outdoor_temp-ref) - delta)))

# Define the range for t
out_temp = np.linspace(-15, 25, 100)

# Calculate alpha(t) for winter and summer
beta_winter = -1
ref_day=22.5*np.ones(100)
alpha_winter_day = alpha(out_temp, ref_day, beta_winter, delta)
ref_winter_night=15*np.ones(100)
alpha_winter_night = alpha(out_temp, ref_winter_night, beta_winter, delta)


fig = plt.figure(figsize=(20, 8))
gs = gridspec.GridSpec(1, 2, figure=fig)
# fig.suptitle(r'Dynamic thermal weight $\alpha(t)$ V.S. fixed thermal weight $\alpha$=0.5',y=0.915)  # Set a title for the whole figure

ax1 = fig.add_subplot(gs[0, 0])  # This adds a subplot that spans the first row entirely
ax1.set_title(r'Winter seasons ($\beta=-1, \delta=17.5^{\circ}\mathrm{C}$)', fontsize=28)
ax1.plot(out_temp, alpha_winter_day, 'b', label=r'$\alpha(t)$ for occupied period', linewidth=4)

ref_night_winter=15*np.ones(100)
alpha_winter_night = alpha(out_temp, ref_night_winter, beta_winter, delta)
ax1.plot(out_temp, alpha_winter_night, 'b:', label=r'$\alpha(t)$ for unoccupied period', linewidth=4)
ax1.legend(loc="upper right", fontsize=20)
ax1.set_xlabel ('Outdoor temperature $t_{{out}}$', fontsize=22)

ax1.set_ylabel (r'$\alpha(t)$', fontsize=20)
ax1.grid(True)

out_temp_summer = np.linspace(-10, 30, 100)
alpha_summer_day = alpha(out_temp_summer, ref_day, beta_summer, delta)
ax2 = fig.add_subplot(gs[0, 1])  # This adds a subplot that spans the first row entirely
ax2.set_title(r'Summer seasons ($\beta=1, \delta=17.5^{\circ}\mathrm{C}$)', fontsize=28)
ax2.plot(out_temp_summer, alpha_summer_day, 'g', label=r'$\alpha(t)$ for occupied period', linewidth=4)

out_temp_summer = np.linspace(-10, 30, 100)
ref_night_summer=30*np.ones(100)
alpha_summer_night = alpha(out_temp_summer, ref_night_summer, beta_summer, delta)

ax2.plot(out_temp_summer, alpha_summer_night, 'g:', label=r'$\alpha(t)$ for unoccupied period', linewidth=4)

ax2.legend(loc="lower right", fontsize=20)
ax2.set_xlabel ('Outdoor temperature $t_{{out}}$', fontsize=22)
ax2.set_ylabel (r'$\alpha(t)$', fontsize=20)
ax2.grid(True)


plt.tight_layout()
plt.savefig('C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\varying\\Thermal_weight.png')
plt.show()