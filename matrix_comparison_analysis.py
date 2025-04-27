import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
import os

# Create output directory for plots
output_dir = "matrix_analysis_plots"
os.makedirs(output_dir, exist_ok=True)

# Define matrices with the exact values from the results
M_dominant = np.array([
    [0.00014777, 0.00018632],
    [0.00018632, 0.00043133]
])

B_dominant = np.array([
    [-0.03564208, -0.01559049],
    [-0.01559049, 0.02676829]
])

K_dominant = np.array([
    [0.01338501, 0.03109889],
    [0.03109889, 0.01019305]
])

M_non_dominant = np.array([
    [0.00031676, 0.00040491],
    [0.00040491, 0.00010771]
])

B_non_dominant = np.array([
    [-0.01691927, 0.00711288],
    [0.00711288, -0.00280769]
])

K_non_dominant = np.array([
    [0.01690897, 0.02341543],
    [0.02341543, 0.0001722]
])

def plot_stiffness_ellipse(K, color, label):
    """
    Create an ellipse patch based on the stiffness matrix eigenvalues and eigenvectors.
    
    Parameters:
    - K: 2x2 stiffness matrix
    - color: color of the ellipse
    - label: label for the legend
    
    Returns:
    - Ellipse patch
    """
    # Calculate eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eig(K)
    
    # Sort eigenvalues and eigenvectors
    idx = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # Calculate angle of rotation (in degrees)
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    
    # Scale factor to make ellipses more visible
    scale_factor = 1000
    
    # Create ellipse patch
    ellipse = Ellipse((0, 0),
                      width=2 * np.abs(eigenvals[0]) * scale_factor,
                      height=2 * np.abs(eigenvals[1]) * scale_factor,
                      angle=angle,
                      facecolor=color,
                      edgecolor='black',
                      label=label,
                      linewidth=2,
                      alpha=0.3)
    
    return ellipse

def plot_matrix_comparison(matrix_d, matrix_nd, title, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot dominant
    sns.heatmap(matrix_d, annot=True, fmt='.6f', cmap='viridis', ax=ax1)
    ax1.set_title('Dominant')
    
    # Plot non-dominant
    sns.heatmap(matrix_nd, annot=True, fmt='.6f', cmap='viridis', ax=ax2)
    ax2.set_title('Non-Dominant')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# Create comparison plots for each matrix
plot_matrix_comparison(M_dominant, M_non_dominant, 'Inertia Matrix (M) Comparison', 'inertia_comparison.png')
plot_matrix_comparison(B_dominant, B_non_dominant, 'Damping Matrix (B) Comparison', 'damping_comparison.png')
plot_matrix_comparison(K_dominant, K_non_dominant, 'Stiffness Matrix (K) Comparison', 'stiffness_comparison.png')

# Set the style with system fonts
plt.style.use('seaborn-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'  # Use DejaVu Sans instead of Arial
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.3

# Plot stiffness ellipses
fig, ax = plt.subplots(figsize=(12, 12))

# Define professional colors
dominant_color = '#2E86C1'  # Blue
non_dominant_color = '#E74C3C'  # Red

# Add ellipses
ax.add_patch(plot_stiffness_ellipse(K_dominant, dominant_color, 'Dominant Arm'))
ax.add_patch(plot_stiffness_ellipse(K_non_dominant, non_dominant_color, 'Non-Dominant Arm'))

# Configure plot with larger limits
limit = 100
ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)

# Customize grid
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_aspect('equal')

# Add axes lines with custom style
ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)

# Customize spines
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# Add labels and title with custom styling
plt.title('Stiffness Ellipses Comparison\nDominant vs Non-Dominant Arm', 
          pad=20, 
          fontsize=16, 
          fontweight='bold')
plt.xlabel('X-axis Stiffness (N/m)', fontsize=12, fontweight='bold')
plt.ylabel('Y-axis Stiffness (N/m)', fontsize=12, fontweight='bold')

# Customize tick labels
ax.tick_params(axis='both', which='major', labelsize=10, width=2, length=6)

# Add custom legend with enhanced styling
legend = plt.legend(loc='upper right', 
                   frameon=True, 
                   fontsize=12, 
                   title='Arm Type',
                   title_fontsize=12,
                   framealpha=0.9,
                   edgecolor='black')
legend.get_frame().set_linewidth(2)

# Add scale factor note with better styling
note_text = f'Note: Ellipses scaled by 1000× for better visualization'
plt.figtext(0.02, 0.02, note_text, 
            fontsize=10, 
            style='italic', 
            bbox=dict(facecolor='white', 
                     edgecolor='gray', 
                     alpha=0.8,
                     boxstyle='round,pad=0.5'))

# Add arrows to show coordinate directions
arrow_props = dict(head_width=3, head_length=5, fc='k', ec='k', linewidth=2)
arrow_length = limit * 0.15
margin = limit * 0.1
ax.arrow(-limit + margin, -limit + margin, arrow_length, 0, **arrow_props)
ax.arrow(-limit + margin, -limit + margin, 0, arrow_length, **arrow_props)
ax.text(-limit + margin + arrow_length/2, -limit + margin - 10, 'X', 
        ha='center', va='top', fontsize=12, fontweight='bold')
ax.text(-limit + margin - 10, -limit + margin + arrow_length/2, 'Y', 
        ha='right', va='center', fontsize=12, fontweight='bold')

# Make sure the plot is properly displayed with tight layout
plt.tight_layout()

# Save with high quality
plt.savefig(os.path.join(output_dir, 'stiffness_ellipses.png'), 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')
plt.close()

# Calculate and print comparative metrics
print(f"\nResults saved to: {output_dir}/")
print("\nComparative Analysis:")
print("=====================")

# Condition numbers
print("\nCondition Numbers:")
print(f"Dominant K: {np.linalg.cond(K_dominant):.2f}")
print(f"Non-Dominant K: {np.linalg.cond(K_non_dominant):.2f}")

# Eigenvalues
print("\nEigenvalues:")
print("Dominant K:", np.linalg.eigvals(K_dominant))
print("Non-Dominant K:", np.linalg.eigvals(K_non_dominant))

# Stiffness isotropy index (ratio of min to max eigenvalue)
def isotropy_index(matrix):
    eigenvals = np.abs(np.linalg.eigvals(matrix))
    return np.min(eigenvals) / np.max(eigenvals)

print("\nIsotropy Indices:")
print(f"Dominant K: {isotropy_index(K_dominant):.4f}")
print(f"Non-Dominant K: {isotropy_index(K_non_dominant):.4f}")

# Matrix norms
print("\nFrobenius Norms:")
print(f"Dominant K: {np.linalg.norm(K_dominant, 'fro'):.4f}")
print(f"Non-Dominant K: {np.linalg.norm(K_non_dominant, 'fro'):.4f}")

# Trace comparison
print("\nMatrix Traces:")
print(f"Dominant K: {np.trace(K_dominant):.4f}")
print(f"Non-Dominant K: {np.trace(K_non_dominant):.4f}")
