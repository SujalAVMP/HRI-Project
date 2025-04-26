import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

# Create output directory for plots if it doesn't exist
output_dir = "arm_position_plots"
os.makedirs(output_dir, exist_ok=True)

# Number of lines to skip (based on CSV structure)
skip_rows = 5

# Get all CSV files in the CSV directory
csv_files = glob.glob("CSV/*.csv")

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

# Process each CSV file
for csv_file in csv_files:
    # Extract file name without extension for plot titles
    file_name = os.path.basename(csv_file).split(".")[0]
    print(f"Processing {file_name}...")

    try:
        # Read CSV file
        df = pd.read_csv(csv_file, skiprows=skip_rows)

        # Drop empty columns
        df = df.dropna(axis=1, how="all")

        # Set column names
        if len(df.columns) == len(column_names):
            df.columns = column_names
        else:
            print(
                f"Warning: Column count mismatch in {file_name}. Expected {len(column_names)}, got {len(df.columns)}."
            )
            # Try to handle this case by assigning as many columns as possible
            df.columns = column_names[: len(df.columns)]

        # Create a figure for arm X and Y positions
        plt.figure(figsize=(12, 6))

        # Get frame data (time)
        frames = df["Frame"].values

        # Plot arm X position
        plt.subplot(1, 2, 1)
        plt.plot(frames, df["Arm_X"].values, "b-")
        plt.title(f"Arm X Position vs Frame - {file_name}")
        plt.xlabel("Frame (200Hz)")
        plt.ylabel("X Position (mm)")
        plt.grid(True)

        # Plot arm Y position
        plt.subplot(1, 2, 2)
        plt.plot(frames, df["Arm_Y"].values, "r-")
        plt.title(f"Arm Y Position vs Frame - {file_name}")
        plt.xlabel("Frame (200Hz)")
        plt.ylabel("Y Position (mm)")
        plt.grid(True)

        plt.tight_layout()

        # Save the plot
        plot_filename = os.path.join(output_dir, f"{file_name}_arm_position.png")
        plt.savefig(plot_filename, dpi=300)
        print(f"Saved plot to {plot_filename}")

        # Close the figure to free memory
        plt.close()

    except Exception as e:
        print(f"Error processing {file_name}: {e}")

# Create a combined plot with all files for comparison
plt.figure(figsize=(15, 10))

# Plot for X positions
plt.subplot(2, 1, 1)
for csv_file in csv_files:
    file_name = os.path.basename(csv_file).split(".")[0]
    try:
        df = pd.read_csv(csv_file, skiprows=skip_rows)
        df = df.dropna(axis=1, how="all")

        if len(df.columns) >= 18:  # Make sure we have enough columns for Arm_X
            # Set column names
            df.columns = column_names[: len(df.columns)]

            # Plot arm X position
            frames = df["Frame"].values
            plt.plot(frames, df["Arm_X"].values, label=file_name)
    except Exception as e:
        print(f"Error including {file_name} in combined plot: {e}")

plt.title("Arm X Position vs Frame - All Files")
plt.xlabel("Frame (200Hz)")
plt.ylabel("X Position (mm)")
plt.grid(True)
plt.legend()

# Plot for Y positions
plt.subplot(2, 1, 2)
for csv_file in csv_files:
    file_name = os.path.basename(csv_file).split(".")[0]
    try:
        df = pd.read_csv(csv_file, skiprows=skip_rows)
        df = df.dropna(axis=1, how="all")

        if len(df.columns) >= 18:  # Make sure we have enough columns for Arm_Y
            # Set column names
            df.columns = column_names[: len(df.columns)]

            # Plot arm Y position
            frames = df["Frame"].values
            plt.plot(frames, df["Arm_Y"].values, label=file_name)
    except Exception as e:
        print(f"Error including {file_name} in combined plot: {e}")

plt.title("Arm Y Position vs Frame - All Files")
plt.xlabel("Frame (200Hz)")
plt.ylabel("Y Position (mm)")
plt.grid(True)
plt.legend()

plt.tight_layout()

# Save the combined plot
combined_plot_filename = os.path.join(output_dir, "all_files_arm_position.png")
plt.savefig(combined_plot_filename, dpi=300)
print(f"Saved combined plot to {combined_plot_filename}")

print("All plots generated successfully!")
