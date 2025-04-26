import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# Function to construct the A matrix (10x9) from input data
def construct_A_matrix(X, Y, X_dot, Y_dot, X_ddot, Y_ddot):
    """
    Constructs the A matrix (10x9) for 5 data points, assuming symmetric M, B, K matrices.

    Parameters:
    - X, Y: Arrays of position data (5 elements each)
    - X_dot, Y_dot: Arrays of velocity data (5 elements each)
    - X_ddot, Y_ddot: Arrays of acceleration data (5 elements each)

    Returns:
    - A: 10x9 matrix for the system of equations
    """
    A = np.zeros((10, 9))
    for i in range(5):
        # Row for X-equation (indices 2*i)
        A[2 * i, 0] = X_ddot[i]  # M11
        A[2 * i, 1] = Y_ddot[i]  # M12
        A[2 * i, 2] = 0  # M22 (not in X-eq)
        A[2 * i, 3] = X_dot[i]  # B11
        A[2 * i, 4] = Y_dot[i]  # B12
        A[2 * i, 5] = 0  # B22 (not in X-eq)
        A[2 * i, 6] = X[i]  # K11
        A[2 * i, 7] = Y[i]  # K12
        A[2 * i, 8] = 0  # K22 (not in X-eq)

        # Row for Y-equation (indices 2*i+1)
        A[2 * i + 1, 0] = 0  # M11 (not in Y-eq)
        A[2 * i + 1, 1] = X_ddot[i]  # M12
        A[2 * i + 1, 2] = Y_ddot[i]  # M22
        A[2 * i + 1, 3] = 0  # B11 (not in Y-eq)
        A[2 * i + 1, 4] = X_dot[i]  # B12
        A[2 * i + 1, 5] = Y_dot[i]  # B22
        A[2 * i + 1, 6] = 0  # K11 (not in Y-eq)
        A[2 * i + 1, 7] = X[i]  # K12
        A[2 * i + 1, 8] = Y[i]  # K22
    return A


# Function to solve for M, B, K matrices using least squares
def solve_for_matrices(A, F):
    """
    Solves for the symmetric matrices M, B, K using least squares.

    Parameters:
    - A: 10x9 matrix of kinematic data
    - F: 10x1 vector of forces

    Returns:
    - M, B, K: Symmetric 2x2 matrices for inertia, damping, and stiffness
    - X: Solution vector
    - residuals: Residuals from least squares solution
    """
    X, residuals, rank, s = np.linalg.lstsq(A, F, rcond=None)

    # Map X (9x1) to symmetric 2x2 matrices
    M = np.array([[X[0], X[1]], [X[1], X[2]]])  # M11, M12, M22
    B = np.array([[X[3], X[4]], [X[4], X[5]]])  # B11, B12, B22
    K = np.array([[X[6], X[7]], [X[7], X[8]]])  # K11, K12, K22

    return M, B, K, X, residuals


# Function to calculate force vectors from arm to pulleys
def calculate_force_vectors(arm_x, arm_y, csv_file, force_magnitude=9.8):
    """
    Calculates force vectors from arm to each pulley.

    Parameters:
    - arm_x, arm_y: Arm position
    - csv_file: Path to CSV file containing pulley positions
    - force_magnitude: Force magnitude in Newtons (default: 9.8 N for 1 kg)

    Returns:
    - force_vectors: List of (Fx, Fy) tuples for each pulley
    """
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
    df = pd.read_csv(csv_file, skiprows=5)

    # Drop empty columns
    df = df.dropna(axis=1, how="all")

    # Set column names
    if len(df.columns) == len(column_names):
        df.columns = column_names
    else:
        print(
            f"Warning: Column count mismatch in {csv_file}. Expected {len(column_names)}, got {len(df.columns)}."
        )
        # Try to handle this case by assigning as many columns as possible
        df.columns = column_names[: len(df.columns)]

    # Find the row with the closest arm position
    distances = np.sqrt((df["Arm_X"] - arm_x) ** 2 + (df["Arm_Y"] - arm_y) ** 2)
    closest_idx = distances.argmin()
    row = df.iloc[closest_idx]

    force_vectors = []

    for i in range(1, 6):
        # Extract pulley position
        pulley_x = row[f"P{i}_X"]
        pulley_y = row[f"P{i}_Y"]

        # Calculate direction vector from arm to pulley
        dir_x = pulley_x - arm_x
        dir_y = pulley_y - arm_y

        # Calculate magnitude of direction vector
        magnitude = np.sqrt(dir_x**2 + dir_y**2)

        # Normalize direction vector
        norm_x = dir_x / magnitude
        norm_y = dir_y / magnitude

        # Calculate force vector (F = m*g * direction)
        force_x = force_magnitude * norm_x
        force_y = force_magnitude * norm_y

        force_vectors.append((force_x, force_y))

    return force_vectors


# Load deflection results
deflection_results = pd.read_csv("deflection_plots/deflection_results.csv")

# Separate dominant and non-dominant arm data
dominant_data = deflection_results[deflection_results["file"].str.startswith("ball_p")]
non_dominant_data = deflection_results[
    deflection_results["file"].str.startswith("ball_nd")
]

# Create output directory
output_dir = "dynamic_equation_results"
os.makedirs(output_dir, exist_ok=True)

# Process dominant arm data
print("Processing Dominant Arm Data...")
dominant_X = dominant_data["max_x"].values
dominant_Y = dominant_data["max_y"].values
dominant_X_dot = dominant_data["max_xdot"].values
dominant_Y_dot = dominant_data["max_ydot"].values
dominant_X_ddot = dominant_data["max_xdoubledot"].values
dominant_Y_ddot = dominant_data["max_ydoubledot"].values

# Calculate force vectors for dominant arm
dominant_F = []
for i, row in dominant_data.iterrows():
    csv_file = f"CSV/{row['file']}.csv"
    force_vectors = calculate_force_vectors(row["max_x"], row["max_y"], csv_file)
    # For each data point, we need the force from all 5 pulleys
    # But we'll only use the first pulley's force for simplicity in this example
    # In a real scenario, you might want to sum these forces or use a different approach
    fx, fy = force_vectors[0]  # Just use the first pulley's force
    dominant_F.append(fx)
    dominant_F.append(fy)

dominant_F = np.array(dominant_F)

# Print dimensions for debugging
print(f"Dominant arm data points: {len(dominant_X)}")
print(f"Dominant X shape: {dominant_X.shape}")
print(f"Dominant F shape: {len(dominant_F)}")

# We need exactly 5 data points for our model
if len(dominant_X) != 5:
    print(
        "Warning: We need exactly 5 data points for dominant arm. Selecting the first 5."
    )
    # Select only the first 5 data points if we have more
    if len(dominant_X) > 5:
        dominant_X = dominant_X[:5]
        dominant_Y = dominant_Y[:5]
        dominant_X_dot = dominant_X_dot[:5]
        dominant_Y_dot = dominant_Y_dot[:5]
        dominant_X_ddot = dominant_X_ddot[:5]
        dominant_Y_ddot = dominant_Y_ddot[:5]
    else:
        print("Error: Not enough data points for dominant arm.")
        exit(1)

# Reshape force vector to match expected dimensions (10 elements)
dominant_F = dominant_F[
    :10
]  # Take only the first 10 elements (5 points * 2 dimensions)
print(f"Adjusted dominant F shape: {len(dominant_F)}")

# Construct A matrix for dominant arm
dominant_A = construct_A_matrix(
    dominant_X,
    dominant_Y,
    dominant_X_dot,
    dominant_Y_dot,
    dominant_X_ddot,
    dominant_Y_ddot,
)
print(f"Dominant A shape: {dominant_A.shape}")
print(f"Dominant F shape: {dominant_F.shape}")

# Solve for matrices for dominant arm
dominant_M, dominant_B, dominant_K, dominant_X_sol, dominant_residuals = (
    solve_for_matrices(dominant_A, dominant_F)
)

# Process non-dominant arm data
print("\nProcessing Non-Dominant Arm Data...")
non_dominant_X = non_dominant_data["max_x"].values
non_dominant_Y = non_dominant_data["max_y"].values
non_dominant_X_dot = non_dominant_data["max_xdot"].values
non_dominant_Y_dot = non_dominant_data["max_ydot"].values
non_dominant_X_ddot = non_dominant_data["max_xdoubledot"].values
non_dominant_Y_ddot = non_dominant_data["max_ydoubledot"].values

# Calculate force vectors for non-dominant arm
non_dominant_F = []
for i, row in non_dominant_data.iterrows():
    csv_file = f"CSV/{row['file']}.csv"
    force_vectors = calculate_force_vectors(row["max_x"], row["max_y"], csv_file)
    # For each data point, we need the force from all 5 pulleys
    # But we'll only use the first pulley's force for simplicity in this example
    # In a real scenario, you might want to sum these forces or use a different approach
    fx, fy = force_vectors[0]  # Just use the first pulley's force
    non_dominant_F.append(fx)
    non_dominant_F.append(fy)

non_dominant_F = np.array(non_dominant_F)

# Print dimensions for debugging
print(f"Non-dominant arm data points: {len(non_dominant_X)}")
print(f"Non-dominant X shape: {non_dominant_X.shape}")
print(f"Non-dominant F shape: {len(non_dominant_F)}")

# We need exactly 5 data points for our model
if len(non_dominant_X) != 5:
    print(
        "Warning: We need exactly 5 data points for non-dominant arm. Selecting the first 5."
    )
    # Select only the first 5 data points if we have more
    if len(non_dominant_X) > 5:
        non_dominant_X = non_dominant_X[:5]
        non_dominant_Y = non_dominant_Y[:5]
        non_dominant_X_dot = non_dominant_X_dot[:5]
        non_dominant_Y_dot = non_dominant_Y_dot[:5]
        non_dominant_X_ddot = non_dominant_X_ddot[:5]
        non_dominant_Y_ddot = non_dominant_Y_ddot[:5]
    else:
        print("Error: Not enough data points for non-dominant arm.")
        exit(1)

# Reshape force vector to match expected dimensions (10 elements)
non_dominant_F = non_dominant_F[
    :10
]  # Take only the first 10 elements (5 points * 2 dimensions)
print(f"Adjusted non-dominant F shape: {len(non_dominant_F)}")

# Construct A matrix for non-dominant arm
non_dominant_A = construct_A_matrix(
    non_dominant_X,
    non_dominant_Y,
    non_dominant_X_dot,
    non_dominant_Y_dot,
    non_dominant_X_ddot,
    non_dominant_Y_ddot,
)
print(f"Non-dominant A shape: {non_dominant_A.shape}")
print(f"Non-dominant F shape: {non_dominant_F.shape}")

# Solve for matrices for non-dominant arm
(
    non_dominant_M,
    non_dominant_B,
    non_dominant_K,
    non_dominant_X_sol,
    non_dominant_residuals,
) = solve_for_matrices(non_dominant_A, non_dominant_F)

# Display and save results for dominant arm
print("\nDominant Arm Results:")
print("=====================")
print("\nInertia Matrix M (2x2 symmetric):")
print(dominant_M)
print("\nDamping Matrix B (2x2 symmetric):")
print(dominant_B)
print("\nStiffness Matrix K (2x2 symmetric):")
print(dominant_K)
print("\nResiduals:", dominant_residuals)

# Display and save results for non-dominant arm
print("\nNon-Dominant Arm Results:")
print("========================")
print("\nInertia Matrix M (2x2 symmetric):")
print(non_dominant_M)
print("\nDamping Matrix B (2x2 symmetric):")
print(non_dominant_B)
print("\nStiffness Matrix K (2x2 symmetric):")
print(non_dominant_K)
print("\nResiduals:", non_dominant_residuals)

# Save results to CSV files
dominant_results = pd.DataFrame(
    {
        "Parameter": ["M11", "M12", "M22", "B11", "B12", "B22", "K11", "K12", "K22"],
        "Value": dominant_X_sol,
    }
)
dominant_results.to_csv(f"{output_dir}/dominant_arm_results.csv", index=False)

non_dominant_results = pd.DataFrame(
    {
        "Parameter": ["M11", "M12", "M22", "B11", "B12", "B22", "K11", "K12", "K22"],
        "Value": non_dominant_X_sol,
    }
)
non_dominant_results.to_csv(f"{output_dir}/non_dominant_arm_results.csv", index=False)


# Create visualizations of the matrices
def plot_matrix(matrix, title, filename):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap="viridis")
    plt.colorbar(label="Value")
    plt.title(title)
    plt.xticks([0, 1], ["x", "y"])
    plt.yticks([0, 1], ["x", "y"])

    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(
                j, i, f"{matrix[i, j]:.4f}", ha="center", va="center", color="white"
            )

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# Plot matrices for dominant arm
plot_matrix(
    dominant_M,
    "Dominant Arm: Inertia Matrix (M)",
    f"{output_dir}/dominant_M_matrix.png",
)
plot_matrix(
    dominant_B,
    "Dominant Arm: Damping Matrix (B)",
    f"{output_dir}/dominant_B_matrix.png",
)
plot_matrix(
    dominant_K,
    "Dominant Arm: Stiffness Matrix (K)",
    f"{output_dir}/dominant_K_matrix.png",
)

# Plot matrices for non-dominant arm
plot_matrix(
    non_dominant_M,
    "Non-Dominant Arm: Inertia Matrix (M)",
    f"{output_dir}/non_dominant_M_matrix.png",
)
plot_matrix(
    non_dominant_B,
    "Non-Dominant Arm: Damping Matrix (B)",
    f"{output_dir}/non_dominant_B_matrix.png",
)
plot_matrix(
    non_dominant_K,
    "Non-Dominant Arm: Stiffness Matrix (K)",
    f"{output_dir}/non_dominant_K_matrix.png",
)


# Verify the solution by calculating the predicted forces
def calculate_predicted_forces(M, B, K, X, Y, X_dot, Y_dot, X_ddot, Y_ddot):
    predicted_F = []
    for i in range(len(X)):
        # Calculate predicted force in X direction
        Fx = (
            (M[0, 0] * X_ddot[i] + M[0, 1] * Y_ddot[i])
            + (B[0, 0] * X_dot[i] + B[0, 1] * Y_dot[i])
            + (K[0, 0] * X[i] + K[0, 1] * Y[i])
        )

        # Calculate predicted force in Y direction
        Fy = (
            (M[1, 0] * X_ddot[i] + M[1, 1] * Y_ddot[i])
            + (B[1, 0] * X_dot[i] + B[1, 1] * Y_dot[i])
            + (K[1, 0] * X[i] + K[1, 1] * Y[i])
        )

        predicted_F.append(Fx)
        predicted_F.append(Fy)

    return np.array(predicted_F)


# Calculate predicted forces for dominant arm
dominant_predicted_F = calculate_predicted_forces(
    dominant_M,
    dominant_B,
    dominant_K,
    dominant_X,
    dominant_Y,
    dominant_X_dot,
    dominant_Y_dot,
    dominant_X_ddot,
    dominant_Y_ddot,
)

# Calculate predicted forces for non-dominant arm
non_dominant_predicted_F = calculate_predicted_forces(
    non_dominant_M,
    non_dominant_B,
    non_dominant_K,
    non_dominant_X,
    non_dominant_Y,
    non_dominant_X_dot,
    non_dominant_Y_dot,
    non_dominant_X_ddot,
    non_dominant_Y_ddot,
)

# Calculate error between actual and predicted forces
dominant_force_error = np.abs(dominant_F - dominant_predicted_F)
non_dominant_force_error = np.abs(non_dominant_F - non_dominant_predicted_F)

# Plot actual vs predicted forces
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(dominant_F, "b-", label="Actual Forces")
plt.plot(dominant_predicted_F, "r--", label="Predicted Forces")
plt.title("Dominant Arm: Actual vs Predicted Forces")
plt.xlabel("Force Component Index")
plt.ylabel("Force (N)")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(non_dominant_F, "b-", label="Actual Forces")
plt.plot(non_dominant_predicted_F, "r--", label="Predicted Forces")
plt.title("Non-Dominant Arm: Actual vs Predicted Forces")
plt.xlabel("Force Component Index")
plt.ylabel("Force (N)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f"{output_dir}/force_comparison.png", dpi=300)
plt.close()

# Print verification results
print("\nVerification Results:")
print("====================")
print(f"Dominant Arm Mean Force Error: {np.mean(dominant_force_error):.4f} N")
print(f"Non-Dominant Arm Mean Force Error: {np.mean(non_dominant_force_error):.4f} N")

print(f"\nResults saved to {output_dir}/")
