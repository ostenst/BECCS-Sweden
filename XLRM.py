import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator, FuncFormatter

# Generate 50 data points within a range to ensure all values are positive
np.random.seed(42)
x = np.random.uniform(0, 2, 50)
y = np.random.uniform(0, 2, 50)
z = x**2 + y**2 + np.random.normal(0, 0.7, 50)  # Paraboloid surface with random perturbation

# Size of the scatter points
point_size = 50  # Adjusted to be twice as large

# Create the figure with two subplots
fig = plt.figure(figsize=(14, 6))

# Original Plot
ax1 = fig.add_subplot(121, projection='3d')
scatter1 = ax1.scatter(x, y, z, c=z, cmap='RdYlGn', marker='o', s=point_size)
# colorbar1 = plt.colorbar(scatter1, ax=ax1, pad=0.1)
# colorbar1.set_label('Z value')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('3D Scatter Plot with Perturbed Surface')

ax1.set_xlim([0, 2])
ax1.set_ylim([0, 2])
ax1.set_zlim([0, max(z) + 1])
ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
ax1.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
ax1.zaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}'.format(y)))
ax1.zaxis.set_major_formatter(FuncFormatter(lambda z, _: '{:.0f}'.format(z)))
ax1.view_init(elev=10, azim=120)

ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False
fig.patch.set_facecolor('white')

# Create the classified plot with different colors and projections
ax2 = fig.add_subplot(122, projection='3d')

# Define colors based on classification
colors = np.where(z > 4, 'green', 'crimson')

# Scatter plot with classification colors
scatter2 = ax2.scatter(x, y, z, c=colors, marker='o', s=point_size)

# Add grey projections for points where z > 4
for xi, yi, zi in zip(x, y, z):
    if zi > 4:
        ax2.plot([xi, xi], [yi, yi], [0, zi], color='green', linestyle='-', linewidth=3, alpha=0.3)

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Classified 3D Scatter Plot with Projections')

ax2.set_xlim([0, 2])
ax2.set_ylim([0, 2])
ax2.set_zlim([0, max(z) + 1])
ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
ax2.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
ax2.zaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}'.format(y)))
ax2.zaxis.set_major_formatter(FuncFormatter(lambda z, _: '{:.0f}'.format(z)))
ax2.view_init(elev=10, azim=120)

ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False
fig.patch.set_facecolor('white')
plt.savefig('XLRM.png', dpi=400, bbox_inches='tight')

plt.show()
