
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import FancyArrowPatch
import matplotlib.colors as mcolors

# Define the CSV file and initial frame
csv_file = "CSV/ball_nd_p1.csv"
initial_frame = 140
force_magnitude = 9.8  # N (1 kg * 9.8 m/s^2)

# Number of lines to skip (based on CSV structure)
skip_rows = 5

# Column names for the data
column_names = [
    "Frame",
    "Sub Frame",
    "P1_X",
    "P1_Y",
    "P1_Z",
    "P2_X",
    "P2_Y",
    "P2_Z",
    "P3_X",
    "P3_Y",
    "P3_Z",
    "P4_X",
    "P4_Y",
    "P4_Z",
    "P5_X",
    "P5_Y",
    "P5_Z",
    "Arm_X",
    "Arm_Y",
    "Arm_Z",
    "P1_VX",
    "P1_VY",
    "P1_VZ",
    "P2_VX",
    "P2_VY",
    "P2_VZ",
    "P3_VX",
    "P3_VY",
    "P3_VZ",
    "P4_VX",
    "P4_VY",
    "P4_VZ",
    "P5_VX",
    "P5_VY",
    "P5_VZ",
    "Arm_VX",
    "Arm_VY",
    "Arm_VZ",
    "P1_AX",
    "P1_AY",
    "P1_AZ",
    "P2_AX",
    "P2_AY",
    "P2_AZ",
    "P3_AX",
    "P3_AY",
    "P3_AZ",
    "P4_AX",
    "P4_AY",
    "P4_AZ",
    "P5_AX",
    "P5_AY",
    "P5_AZ",
    "Arm_AX",
    "Arm_AY",
    "Arm_AZ",
]

# Read CSV file
print(f"Reading {csv_file}...")
df = pd.read_csv(csv_file, skiprows=skip_rows)

# Drop empty columns
df = df.dropna(axis=1, how="all")

# Set column names
if len(df.columns) == len(column_names):
    df.columns = column_names
else:
    print(
        f"Warning: Column count mismatch. Expected {len(column_names)}, got {len(df.columns)}."
    )
    # Try to handle this case by assigning as many columns as possible
    df.columns = column_names[: len(df.columns)]

# Get the row for the initial frame
initial_row_idx = df[df["Frame"] == initial_frame].index
if len(initial_row_idx) == 0:
    print(f"Error: Initial frame {initial_frame} not found in the data.")
    exit(1)

initial_row_idx = initial_row_idx[0]
initial_row = df.loc[initial_row_idx]

# Extract arm position
arm_x = initial_row["Arm_X"]
arm_y = initial_row["Arm_Y"]
arm_z = initial_row["Arm_Z"]

print(
    f"\nArm position at frame {initial_frame}: ({arm_x:.2f}, {arm_y:.2f}, {arm_z:.2f})"
)

# Calculate direction vectors from arm to each pulley
pulley_positions = []
direction_vectors = []
normalized_directions = []
force_vectors = []
distances = []
pulley_names = ["Pulley 1", "Pulley 2", "Pulley 3", "Pulley 4", "Pulley 5"]
colors = list(mcolors.TABLEAU_COLORS)

for i in range(1, 6):
    # Extract pulley position
    pulley_x = initial_row[f"P{i}_X"]
    pulley_y = initial_row[f"P{i}_Y"]
    pulley_z = initial_row[f"P{i}_Z"]

    pulley_positions.append((pulley_x, pulley_y, pulley_z))

    # Calculate direction vector from arm to pulley
    dir_x = pulley_x - arm_x
    dir_y = pulley_y - arm_y
    dir_z = pulley_z - arm_z

    # Store direction vector
    direction_vectors.append((dir_x, dir_y, dir_z))

    # Calculate magnitude of direction vector (distance)
    magnitude = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
    distances.append(magnitude)

    # Normalize direction vector
    norm_x = dir_x / magnitude
    norm_y = dir_y / magnitude
    norm_z = dir_z / magnitude

    normalized_directions.append((norm_x, norm_y, norm_z))

    # Calculate force vector (F = m*g * direction)
    force_x = force_magnitude * norm_x
    force_y = force_magnitude * norm_y
    force_z = force_magnitude * norm_z

    force_vectors.append((force_x, force_y, force_z))

# Print results
print("\nResults:")
print("=" * 100)
print(
    f"{'Pulley':<10} | {'Distance (mm)':<15} | {'Direction Vector':<40} | {'Force Vector (N)':<40}"
)
print("-" * 100)

for i in range(5):
    dir_vec = direction_vectors[i]
    force_vec = force_vectors[i]
    print(
        f"Pulley {i+1:<4} | {distances[i]:15.2f} | ({dir_vec[0]:8.2f}, {dir_vec[1]:8.2f}, {dir_vec[2]:8.2f}) | ({force_vec[0]:8.2f}, {force_vec[1]:8.2f}, {force_vec[2]:8.2f})"
    )

print("=" * 100)

# Create a 2D visualization
plt.figure(figsize=(12, 10))

# Plot arm position
plt.scatter(arm_x, arm_y, color="red", s=150, label="Arm", zorder=10)

# Plot pulley positions and vectors
for i in range(5):
    pulley_x, pulley_y, _ = pulley_positions[i]
    dir_vec = direction_vectors[i]
    force_vec = force_vectors[i]
    color = colors[i % len(colors)]

    # Plot pulley
    plt.scatter(pulley_x, pulley_y, color=color, s=80, label=f"Pulley {i+1}", zorder=5)

    # Draw line from arm to pulley
    plt.plot(
        [arm_x, pulley_x],
        [arm_y, pulley_y],
        color=color,
        linestyle="--",
        alpha=0.5,
        zorder=1,
    )

    # Draw force vector
    # Scale for better visualization
    scale_factor = 20
    arrow_dx = force_vec[0] * scale_factor
    arrow_dy = force_vec[1] * scale_factor

    plt.arrow(
        arm_x,
        arm_y,
        arrow_dx,
        arrow_dy,
        head_width=5,
        head_length=10,
        fc=color,
        ec=color,
        width=1.5,
        zorder=8,
    )

    # Add text labels for force magnitudes near pulley points
    if i in [0, 4]:  # For Pulley 1 and 5, place text above
        y_offset = 20
        va_align = "bottom"
    else:  # For Pulleys 2, 3, and 4, place text below
        y_offset = -20
        va_align = "top"
        
    plt.text(
        pulley_x,
        pulley_y + y_offset,  # Use dynamic offset
        f"F{i+1}: ({force_vec[0]:.2f}, {force_vec[1]:.2f}) N",
        fontsize=9,
        ha="center",
        va=va_align,
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.5"),
        zorder=9,
    )

plt.title("Arm and Pulley Positions with Force Vectors", fontsize=14)
plt.xlabel("X Position (mm)", fontsize=12)
plt.ylabel("Y Position (mm)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.axis("equal")

# Add a legend
plt.legend(loc="upper left", fontsize=10)

# Add annotation explaining the vectors
plt.figtext(
    0.5,
    0.01,
    "Force vectors represent the direction and magnitude of the 9.8N force (1kg weight) applied from each pulley.\n"
    "Vectors are drawn from the arm position in the direction of each pulley.",
    ha="center",
    fontsize=10,
    bbox={"facecolor": "lightyellow", "alpha": 0.9, "pad": 5},
)

# Save the plot
output_dir = "force_vectors"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/force_vectors_ball_nd_p1.png", dpi=300, bbox_inches="tight")

# Create a second plot showing just the force components
plt.figure(figsize=(12, 8))

# Define a professional color palette
bar_color = '#2E86C1'  # Strong blue
grid_color = '#EAECEE'  # Light gray
text_color = '#2C3E50'  # Dark blue-gray

# Customize the style
plt.style.use('seaborn')  # Using just 'seaborn' instead of the specific version
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.color'] = grid_color

# Create bar chart for Fx components
ax1 = plt.subplot(2, 1, 1)
bars_fx = plt.bar(pulley_names, [force_vec[0] for force_vec in force_vectors], 
                 color=bar_color, width=0.6)

# Add value labels on top of each bar for Fx
for bar in bars_fx:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}N',
             ha='center', va='bottom' if height >= 0 else 'top',
             color=text_color)

plt.title('Force Components in X Direction (Fx)', 
         fontsize=14, pad=20, color=text_color, fontweight='bold')
plt.ylabel('Force (N)', fontsize=12, color=text_color)
ax1.set_facecolor('white')
plt.grid(True, axis='y', color=grid_color, linestyle='-', linewidth=0.5)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

# Customize the spines
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('#BDBDBD')
ax1.spines['bottom'].set_color('#BDBDBD')

# Create bar chart for Fy components
ax2 = plt.subplot(2, 1, 2)
bars_fy = plt.bar(pulley_names, [force_vec[1] for force_vec in force_vectors], 
                 color=bar_color, width=0.6)

# Add value labels on top of each bar for Fy
for bar in bars_fy:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}N',
             ha='center', va='bottom' if height >= 0 else 'top',
             color=text_color)

plt.title('Force Components in Y Direction (Fy)', 
         fontsize=14, pad=20, color=text_color, fontweight='bold')
plt.ylabel('Force (N)', fontsize=12, color=text_color)
ax2.set_facecolor('white')
plt.grid(True, axis='y', color=grid_color, linestyle='-', linewidth=0.5)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

# Customize the spines
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#BDBDBD')
ax2.spines['bottom'].set_color('#BDBDBD')

# Adjust layout and add main title
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle('Force Components Analysis', 
            fontsize=16, color=text_color, fontweight='bold', y=0.98)

# Save the force components plot with higher DPI and tight layout
plt.savefig(f"{output_dir}/force_components_ball_nd_p1.png", 
            dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()

print(f"\nPlots saved to {output_dir}/")

# Create a summary table for Fx and Fy only
print("\nForce Components (Fx, Fy):")
print("=" * 60)
print(f"{'Pulley':<10} | {'Fx (N)':<10} | {'Fy (N)':<10} | {'Magnitude (N)':<15}")
print("-" * 60)

for i in range(5):
    force_vec = force_vectors[i]
    force_magnitude_2d = np.sqrt(force_vec[0] ** 2 + force_vec[1] ** 2)
    print(
        f"Pulley {i+1:<4} | {force_vec[0]:10.2f} | {force_vec[1]:10.2f} | {force_magnitude_2d:15.2f}"
    )

print("=" * 60)

# Calculate the net force
net_fx = sum(force_vec[0] for force_vec in force_vectors)
net_fy = sum(force_vec[1] for force_vec in force_vectors)
net_magnitude = np.sqrt(net_fx**2 + net_fy**2)

print(f"\nNet Force:")
print(f"Net Fx: {net_fx:.2f} N")
print(f"Net Fy: {net_fy:.2f} N")
print(f"Net Magnitude: {net_magnitude:.2f} N")

# Create a plot showing the net force
plt.figure(figsize=(10, 10))

# Plot arm position
plt.scatter(arm_x, arm_y, color="red", s=150, label="Arm", zorder=10)

# Plot individual force vectors
for i in range(5):
    force_vec = force_vectors[i]
    color = colors[i % len(colors)]

    # Scale for better visualization
    scale_factor = 20
    arrow_dx = force_vec[0] * scale_factor
    arrow_dy = force_vec[1] * scale_factor

    plt.arrow(
        arm_x,
        arm_y,
        arrow_dx,
        arrow_dy,
        head_width=5,
        head_length=10,
        fc=color,
        ec=color,
        width=1.5,
        alpha=0.5,
        zorder=5,
        label=f"F{i+1}",
    )

plt.title(f"Force Vectors at Frame {initial_frame}", fontsize=14)
plt.xlabel("X Position (mm)", fontsize=12)
plt.ylabel("Y Position (mm)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.axis("equal")

# Add a legend
plt.legend(loc="upper left", fontsize=10)

# Add annotation explaining the vectors
plt.figtext(
    0.5,
    0.01,
    "Individual force vectors showing the direction and magnitude of forces from each pulley.",
    ha="center",
    fontsize=10,
    bbox={"facecolor": "lightyellow", "alpha": 0.9, "pad": 5},
)

# Save the net force plot
plt.savefig(f"{output_dir}/net_force_ball_nd_p1.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Net force plot saved to {output_dir}/net_force_ball_nd_p1.png")
