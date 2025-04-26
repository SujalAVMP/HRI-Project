import numpy as np

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
        A[2*i, 0] = X_ddot[i]  # M11
        A[2*i, 1] = Y_ddot[i]  # M12
        A[2*i, 2] = 0          # M22 (not in X-eq)
        A[2*i, 3] = X_dot[i]   # B11
        A[2*i, 4] = Y_dot[i]   # B12
        A[2*i, 5] = 0          # B22 (not in X-eq)
        A[2*i, 6] = X[i]       # K11
        A[2*i, 7] = Y[i]       # K12
        A[2*i, 8] = 0          # K22 (not in X-eq)
        
        # Row for Y-equation (indices 2*i+1)
        A[2*i+1, 0] = 0          # M11 (not in Y-eq)
        A[2*i+1, 1] = X_ddot[i]  # M12
        A[2*i+1, 2] = Y_ddot[i]  # M22
        A[2*i+1, 3] = 0          # B11 (not in Y-eq)
        A[2*i+1, 4] = X_dot[i]   # B12
        A[2*i+1, 5] = Y_dot[i]   # B22
        A[2*i+1, 6] = 0          # K11 (not in Y-eq)
        A[2*i+1, 7] = X[i]       # K12
        A[2*i+1, 8] = Y[i]       # K22
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
    """
    X, residuals, rank, s = np.linalg.lstsq(A, F, rcond=None)
    # Map X (9x1) to symmetric 2x2 matrices
    M = np.array([[X[0], X[1]], [X[1], X[2]]])  # M11, M12, M22
    B = np.array([[X[3], X[4]], [X[4], X[5]]])  # B11, B12, B22
    K = np.array([[X[6], X[7]], [X[7], X[8]]])  # K11, K12, K22
    return M, B, K, X

# Input data for 5 cases
num_points = 5
X = []
Y = []
X_dot = []
Y_dot = []
X_ddot = []
Y_ddot = []
F = []

print("Enter the data for 5 points:")
for i in range(num_points):
    print(f"\nData Point {i+1}:")
    X.append(float(input(f"Enter X position for point {i+1}: ")))
    Y.append(float(input(f"Enter Y position for point {i+1}: ")))
    X_dot.append(float(input(f"Enter X velocity (X') for point {i+1}: ")))
    Y_dot.append(float(input(f"Enter Y velocity (Y') for point {i+1}: ")))
    X_ddot.append(float(input(f"Enter X acceleration (X'') for point {i+1}: ")))
    Y_ddot.append(float(input(f"Enter Y acceleration (Y'') for point {i+1}: ")))

# Input force data for 5 points (10 values for X and Y directions)
print("\nEnter the force data for 5 points (X and Y components):")
for i in range(num_points):
    F_x = float(input(f"Enter Force in X-direction for point {i+1}: "))
    F_y = float(input(f"Enter Force in Y-direction for point {i+1}: "))
    F.append(F_x)
    F.append(F_y)

# Convert lists to numpy arrays
X = np.array(X)
Y = np.array(Y)
X_dot = np.array(X_dot)
Y_dot = np.array(Y_dot)
X_ddot = np.array(X_ddot)
Y_ddot = np.array(Y_ddot)
F = np.array(F)  # 10x1 vector

# Construct A matrix (10x9)
A = construct_A_matrix(X, Y, X_dot, Y_dot, X_ddot, Y_ddot)

# Solve for matrices M, B, K and get solution vector X
M, B, K, X = solve_for_matrices(A, F)

# Display results
print("\nA Matrix (10x9):")
print(A)
print("\nForce Vector F (10x1):")
print(F)
print("\nSolution Vector X (9x1):")
print(X)
print("\nInertia Matrix M (2x2 symmetric):")
print(M)
print("\nDamping Matrix B (2x2 symmetric):")
print(B)
print("\nStiffness Matrix K (2x2 symmetric):")
print(K)
