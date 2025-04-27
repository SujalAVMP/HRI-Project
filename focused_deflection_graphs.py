import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

# Create output directory for focused plots
output_dir = "focused_deflection_plots"
os.makedirs(output_dir, exist_ok=True)

# Load deflection results to get the max deflection frames
deflection_results = pd.read_csv("deflection_plots/deflection_results.csv")

# Number of lines to skip (based on CSV structure)
skip_rows = 5

# Column names (using the same as in your other scripts)
column_names = [
    "Frame", "Sub Frame", 
    "P1_X", "P1_Y", "P1_Z", "P2_X", "P2_Y", "P2_Z",
    "P3_X", "P3_Y", "P3_Z", "P4_X", "P4_Y", "P4_Z",
    "P5_X", "P5_Y", "P5_Z", "Arm_X", "Arm_Y", "Arm_Z",
    "P1_VX", "P1_VY", "P1_VZ", "P2_VX", "P2_VY", "P2_VZ",
    "P3_VX", "P3_VY", "P3_VZ", "P4_VX", "P4_VY", "P4_VZ",
    "P5_VX", "P5_VY", "P5_VZ", "Arm_VX", "Arm_VY", "Arm_VZ",
    "P1_AX", "P1_AY", "P1_AZ", "P2_AX", "P2_AY", "P2_AZ",
    "P3_AX", "P3_AY", "P3_AZ", "P4_AX", "P4_AY", "P4_AZ",
    "P5_AX", "P5_AY", "P5_AZ", "Arm_AX", "Arm_AY", "Arm_AZ"
]

def process_file(csv_file, initial_frame, max_deflection_frame):
    try:
        # Read CSV file
        df = pd.read_csv(csv_file, skiprows=skip_rows)
        df = df.dropna(axis=1, how='all')
        
        # Convert frames to time (seconds)
        def frame_to_time(frame):
            return frame / 200.0  # 200Hz sampling rate
        
        # Create figure with professional style
        plt.style.use('seaborn')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Define colors and styles
        main_color = '#2E86C1'  # Strong blue for main line
        initial_color = '#27AE60'  # Green for initial point
        max_color = '#E74C3C'  # Red for max deflection point
        grid_color = '#EAECEE'  # Light gray for grid
        text_color = '#2C3E50'  # Dark blue-gray for text
        
        # Ensure we have the right number of columns
        if len(df.columns) >= len(column_names):
            df.columns = column_names[:len(df.columns)]
        
        # Calculate the window start and end frames
        window_start = max(initial_frame - 50, 0)  # 50 frames before initial frame
        window_end = window_start + 300  # 300 frame window
        
        # Filter data for our window
        window_data = df[(df['Frame'] >= window_start) & (df['Frame'] <= window_end)]
        
        # Get initial position
        initial_data = df[df['Frame'] == initial_frame]
        if initial_data.empty:
            raise ValueError(f"Initial frame {initial_frame} not found in data")
        x0 = float(initial_data['Arm_X'].values[0])
        y0 = float(initial_data['Arm_Y'].values[0])
        
        # Get max deflection position
        max_data = df[df['Frame'] == max_deflection_frame]
        if max_data.empty:
            raise ValueError(f"Max deflection frame {max_deflection_frame} not found in data")
        max_x = float(max_data['Arm_X'].values[0])
        max_y = float(max_data['Arm_Y'].values[0])
        
        # Convert Frame and Arm_X/Y to numpy arrays and calculate time
        frames = window_data['Frame'].to_numpy()
        times = [frame_to_time(f) for f in frames]
        arm_x = window_data['Arm_X'].to_numpy()
        arm_y = window_data['Arm_Y'].to_numpy()
        
        initial_time = frame_to_time(initial_frame)
        max_time = frame_to_time(max_deflection_frame)
        
        # Plot X position with enhanced styling
        ax1.plot(times, arm_x, color=main_color, linewidth=2)
        ax1.set_title('Arm X Position vs Time', fontsize=14, pad=20, color=text_color, fontweight='bold')
        ax1.set_xlabel('Time (seconds)', fontsize=12, color=text_color)
        ax1.set_ylabel('X Position (mm)', fontsize=12, color=text_color)
        ax1.grid(True, color=grid_color, linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Mark important points on X plot
        ax1.plot(initial_time, x0, 'o', color=initial_color, markersize=10, 
                label=f'Initial Position (t={initial_time:.2f}s)')
        ax1.plot(max_time, max_x, 'o', color=max_color, markersize=10, 
                label=f'Max Deflection (t={max_time:.2f}s)')
        ax1.axvline(x=initial_time, color=initial_color, linestyle='--', alpha=0.3)
        ax1.axvline(x=max_time, color=max_color, linestyle='--', alpha=0.3)
        ax1.legend(fontsize=10, loc='best', framealpha=0.9)
        
        # Customize ax1 spines
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_color('#BDBDBD')
        ax1.spines['bottom'].set_color('#BDBDBD')
        
        # Plot Y position with enhanced styling
        ax2.plot(times, arm_y, color=main_color, linewidth=2)
        ax2.set_title('Arm Y Position vs Time', fontsize=14, pad=20, color=text_color, fontweight='bold')
        ax2.set_xlabel('Time (seconds)', fontsize=12, color=text_color)
        ax2.set_ylabel('Y Position (mm)', fontsize=12, color=text_color)
        ax2.grid(True, color=grid_color, linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Mark important points on Y plot
        ax2.plot(initial_time, y0, 'o', color=initial_color, markersize=10, 
                label=f'Initial Position (t={initial_time:.2f}s)')
        ax2.plot(max_time, max_y, 'o', color=max_color, markersize=10, 
                label=f'Max Deflection (t={max_time:.2f}s)')
        ax2.axvline(x=initial_time, color=initial_color, linestyle='--', alpha=0.3)
        ax2.axvline(x=max_time, color=max_color, linestyle='--', alpha=0.3)
        ax2.legend(fontsize=10, loc='best', framealpha=0.9)
        
        # Customize ax2 spines
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color('#BDBDBD')
        ax2.spines['bottom'].set_color('#BDBDBD')
        
        # Calculate deflection
        deflection = np.sqrt((max_x - x0)**2 + (max_y - y0)**2)
        
        # Add deflection information with enhanced styling
        info_text = (
            f'Maximum Deflection: {deflection:.2f} mm\n'
            f'Initial Position: ({x0:.2f}, {y0:.2f}) mm\n'
            f'Max Deflection Position: ({max_x:.2f}, {max_y:.2f}) mm\n'
            f'Time to Max Deflection: {(max_time - initial_time):.3f} s'
        )
        
        plt.figtext(0.5, 0.02, info_text,
                   ha='center', va='bottom',
                   bbox=dict(facecolor='#F8F9F9', edgecolor='#BDC3C7', 
                           alpha=0.9, boxstyle='round,pad=1'),
                   fontsize=10, color=text_color)
        
        # Add main title
        fig.suptitle('Arm Position Analysis', 
                    fontsize=16, color=text_color, 
                    fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save the plot with high quality
        file_name = os.path.basename(csv_file).split('.')[0]
        plot_filename = os.path.join(output_dir, f"{file_name}_focused_deflection.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Generated focused plot for {file_name}")
        
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")

# Process each file in the deflection results
for _, row in deflection_results.iterrows():
    csv_file = f"CSV/{row['file']}.csv"
    if 'initial_frame' in row and 'max_deflection_frame' in row:
        process_file(csv_file, int(row['initial_frame']), int(row['max_deflection_frame']))
    else:
        print(f"Missing required frame data for {csv_file}")

print(f"\nAll focused deflection plots have been saved to {output_dir}/")
