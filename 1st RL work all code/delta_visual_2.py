import numpy as np
import matplotlib.pyplot as plt

# Function to calculate alpha(t) based on the formula
def calculate_alpha(t_out, t_ref, beta, delta):
    return 1 / (1 + np.exp(beta * (np.abs(t_out - t_ref) - delta)))

# Parameters
t_out = np.linspace(0, 35, 500)  # Absolute difference range
t_ref = 0  # Reference temperature difference is 0 after subtraction
beta_winter = -1  # Beta value for winter
beta_summer = 1   # Beta value for summer
delta_values = [0, 17.5, 35]  # Different delta values to test
colors = ['green', 'cyan', 'blue']  # Colors for different delta values

# Plotting
plt.figure(figsize=(10, 7))

# Plot for winter season
for delta, color in zip(delta_values, colors):
    alpha_winter = calculate_alpha(t_out, t_ref, beta_winter, delta)
    plt.plot(t_out, alpha_winter, label=r'$\alpha(t)$ for winter $(\delta = {}$)'.format(delta), color=color, linestyle='-', linewidth=2)

# Plot for summer season
for delta, color in zip(delta_values, colors):
    alpha_summer = calculate_alpha(t_out, t_ref, beta_summer, delta)
    plt.plot(t_out, alpha_summer, label=r'$\alpha(t)$ for summer $(\delta = {}$)'.format(delta), color=color, linestyle='--', linewidth=2)

# Vertical line for delta = 17.5
plt.axvline(x=17.5, color='black', linestyle='--')

# Titles and labels
plt.title(r'Thermal weight $\alpha(t)$ for different $\delta$ values in winter and summer seasons')
plt.xlabel(r'The absolute difference between reference temperature and outdoor temperature: $|t_{out} - t_{ref}|$')
plt.ylabel(r'Thermal weight $\alpha(t)$')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
