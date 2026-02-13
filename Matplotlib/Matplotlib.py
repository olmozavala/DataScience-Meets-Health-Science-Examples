# %% Matplotlib Examples
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

# Scatter Plot
plt.scatter(x, y_sin)
plt.scatter(x, y_cos)
plt.show()

# %%

# Bar Plot
plt.bar(x, y_sin)
plt.bar(x, y_cos)
plt.show()

# %%
# 3D plot create z depending on x and y

z = np.cos(x) + np.cos(y_cos)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y_sin, z)
plt.show()

# %%
# Animation Example - Animated sine wave

# Set up the figure and axis
fig, ax = plt.subplots()
x_anim = np.linspace(0, 2 * np.pi, 100)
line, = ax.plot(x_anim, np.sin(x_anim))
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Animated Sine Wave')
ax.grid(True)

def animate(frame: int) -> tuple:
    """
    Update the plot for each animation frame.
    
    Parameters:
    -----------
    frame : int
        Current frame number
        
    Returns:
    --------
    tuple
        Tuple of line objects to update
    """
    # Shift the sine wave based on frame number
    y = np.sin(x_anim + frame * 0.1)
    line.set_ydata(y)
    return line,

# Create animation
anim = animation.FuncAnimation(
    fig, 
    animate, 
    frames=100,  # Number of frames
    interval=50,  # Delay between frames in milliseconds
    blit=True  # Only redraw changed parts
)

plt.show()
