import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

# Create output directory for plots if it doesn't exist
output_dir = "deflection_plots"
os.makedirs(output_dir, exist_ok=True)

# Number of lines to skip (based on CSV structure)
skip_rows = 5

# Get all CSV files in the CSV directory
csv_files = glob.glob("CSV/*.csv")

# Column names for the data
column_names = [
    "Frame", "Sub Frame",
    "P1_X", "P1_Y", "P1_Z",
    "P2_X", "P2_Y", "P2_Z",
    "P3_X", "P3_Y", "P3_Z",
    "P4_X", "P4_Y", "P4_Z",
    "P5_X", "P5_Y", "P5_Z",
    "Arm_X", "Arm_Y", "Arm_Z",
    "P1_VX", "P1_VY", "P1_VZ",
    "P2_VX", "P2_VY", "P2_VZ",
    "P3_VX", "P3_VY", "P3_VZ",
    "P4_VX", "P4_VY", "P4_VZ",
    "P5_VX", "P5_VY", "P5_VZ",
    "Arm_VX", "Arm_VY", "Arm_VZ",
    "P1_AX", "P1_AY", "P1_AZ",
    "P2_AX", "P2_AY", "P2_AZ",
    "P3_AX", "P3_AY", "P3_AZ",
    "P4_AX", "P4_AY", "P4_AZ",
    "P5_AX", "P5_AY", "P5_AZ",
    "Arm_AX", "Arm_AY", "Arm_AZ",
]

# Initial frames for each file
initial_frames = {
    "ball_nd_p1": 140,
    "ball_nd_p2": 140,
    "ball_nd_p3": 310,
    "ball_nd_p4": 60,
    "ball_nd_p5": 960,
    "ball_p1": 110,
    "ball_p2": 85,
    "ball_p3": 35,
    "ball_p4": 95,
    "ball_p5": 60
}

# Results table
results = []

# Process each CSV file
for csv_file in csv_files:
    # Extract file name without extension for plot titles
    file_name = os.path.basename(csv_file).split(".")[0]
    print(f"Processing {file_name}...")
    
    if file_name not in initial_frames:
        print(f"Warning: No initial frame defined for {file_name}. Skipping.")
        continue
    
    initial_frame = initial_frames[file_name]
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_file, skiprows=skip_rows)
        
        # Drop empty columns
        df = df.dropna(axis=1, how="all")
        
        # Set column names
        if len(df.columns) == len(column_names):
            df.columns = column_names
        else:
            print(f"Warning: Column count mismatch in {file_name}. Expected {len(column_names)}, got {len(df.columns)}.")
            # Try to handle this case by assigning as many columns as possible
            df.columns = column_names[: len(df.columns)]
        
        # Get the initial position
        initial_row_idx = df[df['Frame'] == initial_frame].index
        if len(initial_row_idx) == 0:
            print(f"Error: Initial frame {initial_frame} not found in {file_name}. Skipping.")
            continue
            
        initial_row_idx = initial_row_idx[0]
        x0 = df.loc[initial_row_idx, 'Arm_X']
        y0 = df.loc[initial_row_idx, 'Arm_Y']
        
        # Calculate deflection for the next 200 frames
        end_frame = min(initial_frame + 200, df['Frame'].max())
        search_range = df[(df['Frame'] >= initial_frame) & (df['Frame'] <= end_frame)]
        
        # Calculate Euclidean distance (deflection) from initial position
        search_range['deflection'] = np.sqrt(
            (search_range['Arm_X'] - x0)**2 + 
            (search_range['Arm_Y'] - y0)**2
        )
        
        # Find the frame with maximum deflection
        max_deflection_idx = search_range['deflection'].idxmax()
        max_deflection_frame = df.loc[max_deflection_idx, 'Frame']
        max_deflection = search_range['deflection'].max()
        
        # Get the corresponding values
        max_x = df.loc[max_deflection_idx, 'Arm_X']
        max_y = df.loc[max_deflection_idx, 'Arm_Y']
        max_xdot = df.loc[max_deflection_idx, 'Arm_VX']
        max_ydot = df.loc[max_deflection_idx, 'Arm_VY']
        max_xdoubledot = df.loc[max_deflection_idx, 'Arm_AX']
        max_ydoubledot = df.loc[max_deflection_idx, 'Arm_AY']
        
        # Store results
        results.append({
            'file': file_name,
            'initial_frame': initial_frame,
            'max_deflection_frame': max_deflection_frame,
            'max_deflection': max_deflection,
            'x0': x0,
            'y0': y0,
            'max_x': max_x,
            'max_y': max_y,
            'max_xdot': max_xdot,
            'max_ydot': max_ydot,
            'max_xdoubledot': max_xdoubledot,
            'max_ydoubledot': max_ydoubledot
        })
        
        # Create a figure for arm X and Y positions with initial and max points labeled
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Get frame data (time)
        frames = df['Frame'].values
        
        # Plot arm X position
        ax1.plot(frames, df['Arm_X'].values, 'b-')
        ax1.set_title(f'Arm X Position vs Frame - {file_name}')
        ax1.set_xlabel('Frame (200Hz)')
        ax1.set_ylabel('X Position (mm)')
        ax1.grid(True)
        
        # Mark initial and max points on X plot
        ax1.plot(initial_frame, x0, 'go', markersize=8, label=f'Initial Frame ({initial_frame})')
        ax1.plot(max_deflection_frame, max_x, 'ro', markersize=8, label=f'Max Deflection Frame ({max_deflection_frame})')
        ax1.legend()
        
        # Add vertical lines for initial and max frames
        ax1.axvline(x=initial_frame, color='g', linestyle='--', alpha=0.5)
        ax1.axvline(x=max_deflection_frame, color='r', linestyle='--', alpha=0.5)
        
        # Plot arm Y position
        ax2.plot(frames, df['Arm_Y'].values, 'b-')
        ax2.set_title(f'Arm Y Position vs Frame - {file_name}')
        ax2.set_xlabel('Frame (200Hz)')
        ax2.set_ylabel('Y Position (mm)')
        ax2.grid(True)
        
        # Mark initial and max points on Y plot
        ax2.plot(initial_frame, y0, 'go', markersize=8, label=f'Initial Frame ({initial_frame})')
        ax2.plot(max_deflection_frame, max_y, 'ro', markersize=8, label=f'Max Deflection Frame ({max_deflection_frame})')
        ax2.legend()
        
        # Add vertical lines for initial and max frames
        ax2.axvline(x=initial_frame, color='g', linestyle='--', alpha=0.5)
        ax2.axvline(x=max_deflection_frame, color='r', linestyle='--', alpha=0.5)
        
        # Add deflection information to the plot
        plt.figtext(0.5, 0.01, 
                   f"Max Deflection: {max_deflection:.2f} mm at Frame {max_deflection_frame}\n" +
                   f"Initial Position: ({x0:.2f}, {y0:.2f}), Max Position: ({max_x:.2f}, {max_y:.2f})",
                   ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save the plot
        plot_filename = os.path.join(output_dir, f"{file_name}_deflection.png")
        plt.savefig(plot_filename, dpi=300)
        print(f"Saved plot to {plot_filename}")
        
        # Close the figure to free memory
        plt.close()
        
        # Create a plot showing the deflection over time
        plt.figure(figsize=(10, 6))
        plt.plot(search_range['Frame'], search_range['deflection'], 'b-')
        plt.title(f'Arm Deflection vs Frame - {file_name}')
        plt.xlabel('Frame (200Hz)')
        plt.ylabel('Deflection (mm)')
        plt.grid(True)
        
        # Mark initial and max points
        plt.plot(initial_frame, 0, 'go', markersize=8, label=f'Initial Frame ({initial_frame})')
        plt.plot(max_deflection_frame, max_deflection, 'ro', markersize=8, label=f'Max Deflection Frame ({max_deflection_frame})')
        plt.legend()
        
        # Add vertical lines for initial and max frames
        plt.axvline(x=initial_frame, color='g', linestyle='--', alpha=0.5)
        plt.axvline(x=max_deflection_frame, color='r', linestyle='--', alpha=0.5)
        
        # Add deflection information to the plot
        plt.figtext(0.5, 0.01, 
                   f"Max Deflection: {max_deflection:.2f} mm at Frame {max_deflection_frame}",
                   ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save the plot
        deflection_plot_filename = os.path.join(output_dir, f"{file_name}_deflection_curve.png")
        plt.savefig(deflection_plot_filename, dpi=300)
        print(f"Saved deflection curve plot to {deflection_plot_filename}")
        
        # Close the figure to free memory
        plt.close()
        
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

# Create a results DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_csv = os.path.join(output_dir, "deflection_results.csv")
results_df.to_csv(results_csv, index=False)
print(f"Saved results to {results_csv}")

# Print results table
print("\nResults Summary:")
print("=" * 100)
print(f"{'File':<12} | {'Initial Frame':<13} | {'Max Frame':<9} | {'Max Deflection':<14} | {'X':<8} | {'Y':<8} | {'X Dot':<8} | {'Y Dot':<8} | {'X DDot':<8} | {'Y DDot':<8}")
print("-" * 100)
for r in results:
    print(f"{r['file']:<12} | {r['initial_frame']:<13} | {r['max_deflection_frame']:<9} | {r['max_deflection']:<14.2f} | {r['max_x']:<8.2f} | {r['max_y']:<8.2f} | {r['max_xdot']:<8.2f} | {r['max_ydot']:<8.2f} | {r['max_xdoubledot']:<8.2f} | {r['max_ydoubledot']:<8.2f}")
print("=" * 100)

print("\nAll deflection analysis completed successfully!")
