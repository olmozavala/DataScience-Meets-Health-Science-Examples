# %% Matplotlib Examples
import numpy as np
import matplotlib.pyplot as plt

# %% Basic Plots

# Plot sine, cosine, and tangent functions

x = np.linspace(0, 2 * np.pi, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.plot(x, y_sin, label='sin(x)')
plt.plot(x, y_cos, label='cos(x)')
plt.legend()
plt.show()


# %%
