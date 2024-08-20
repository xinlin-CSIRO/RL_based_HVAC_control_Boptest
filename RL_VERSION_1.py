import numpy as np
import matplotlib.pyplot as plt

# Define the original optimized function
def optimized_outdoor_diff(x, beta=1, delta=17.5):
    return 1 / (1 + np.exp(-beta * (x - delta)))

# Define the reversed function
def reversed_optimized_outdoor_diff(x, beta=1, delta=17.5):
    return 1 / (1 + np.exp(beta * (x - delta)))

# Generate x values
x = np.linspace(0, 35, 400)

# Calculate y values for both functions
y_original = optimized_outdoor_diff(x)
y_reversed = reversed_optimized_outdoor_diff(x)

# Plotting both functions
plt.figure(figsize=(10, 6))
plt.plot(x, y_original, label=r'$\alpha$(t) for winter seasons ($\beta$=-1, $\delta$=17.5)', color='blue')
plt.plot(x, y_reversed, label=r'$\alpha$(t) for summer seasons ($\beta$= 1, $\delta$=17.5)', color='green')
plt.title(r'Thermal weight $\alpha$(t) for winter and summer seasons')
plt.xlabel('The absolute difference between reference temperature and outdoor temperature: |t_$_{out}$-t_$_{ref}$|')
plt.ylabel(r'Thermal weight $\alpha$(t)')
plt.axvline(x=17.5, color='black', linestyle='--')
plt.legend()
plt.grid(True)
plt.show()
