import numpy as np
import matplotlib.pyplot as plt
import os

# Function to simulate congestion data with or without a discernible knee
def generate_congestion_data(num_points=100, max_latency=60, noise_level=0.2, knee_speed=40, clear_knee=True):
    # Generate internet speeds (linear increase from 1 to 60 Mbps)
    speeds = np.linspace(1, 60, num_points)
    
    if clear_knee:
        # Create a base congestion curve with a soft knee at knee_speed
        latency = np.piecewise(
            speeds,
            [speeds < knee_speed, speeds >= knee_speed],
            [lambda x: 0.01 * x**2, lambda x: 0.001 * (x - knee_speed)**3 + 25]
        )
    else:
        # Generate a gradual curve with a slight increase in latency (quadratic for slow rise)
        latency = 0.0015 * speeds**2 + 5  # A slight upward quadratic curve

    # Add random fluctuations to simulate real-world randomness
    random_fluctuations = np.random.normal(0, noise_level * 10, num_points)
    latency_noisy = latency + random_fluctuations

    # Ensure latency does not exceed the maximum latency
    latency_noisy = np.minimum(latency_noisy, max_latency)
    
    # Ensure latency is never below 0
    latency_noisy = np.maximum(latency_noisy, 0)

    return speeds, latency_noisy

# Function to simulate proportional trends with not very steep gradient and noise
def generate_proportional_trend(num_points=100, slope=0.3, intercept=5, max_latency_diff=15, noise_level=0.2):
    # Generate internet speeds (linear increase from 1 to 60 Mbps)
    speeds = np.linspace(1, 60, num_points)
    
    # Linear increase in latency with a slow gradient
    latency = slope * speeds + intercept

    # Ensure the latency difference between highest and lowest speed does not exceed max_latency_diff
    max_latency = intercept + slope * 60
    if max_latency - intercept > max_latency_diff:
        slope = max_latency_diff / 60  # Adjust the slope to ensure the max difference is within bounds
        latency = slope * speeds + intercept

    # Add random noise
    random_fluctuations = np.random.normal(0, noise_level * 5, num_points)
    latency_noisy = latency + random_fluctuations

    # Ensure latency is never below 0 and doesn't exceed max_latency
    latency_noisy = np.maximum(latency_noisy, 0)
    latency_noisy = np.minimum(latency_noisy, max_latency)

    return speeds, latency_noisy

# Function to save the dataset as an ASCII file
def save_dataset_as_ascii(speeds, latency, filename):
    # Create a dataset with two columns: speeds and latency
    data = np.column_stack((speeds, latency))
    np.savetxt(filename, data, header="Speed (Mbps)   Latency (ms)", fmt="%10.4f")

# Function to plot and save each dataset
def plot_and_save_dataset(speeds, latency, plot_title, dataset_num, datasets_dir, plots_dir):
    # Plot the dataset
    plt.figure(figsize=(8, 5))
    plt.scatter(speeds, latency, alpha=0.6, label='Latency')

    # Plot a smoothed line for each dataset
    z = np.polyfit(speeds, latency, 3)  # Fit a 3rd-degree polynomial to show the general trend
    p = np.poly1d(z)
    plt.plot(speeds, p(speeds), linewidth=2, color='red', label='Smoothed Curve')

    # Labeling the plot
    plt.title(plot_title)
    plt.xlabel('Average download speed (Mbps)')
    plt.ylabel('Access latency (ms)')
    plt.grid(True)
    plt.legend()

    # Save the plot and dataset into dedicated folders
    plot_filename = os.path.join(plots_dir, f'plot_{dataset_num}.png')
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to avoid overwriting
    print(f"Plot saved as {plot_filename}")

    # Save the dataset as an ASCII file
    dataset_filename = os.path.join(datasets_dir, f'dataset_{dataset_num}.txt')
    save_dataset_as_ascii(speeds, latency, dataset_filename)
    print(f"Dataset saved as {dataset_filename}")

# Function to generate multiple datasets and save plots and ASCII files
def plot_and_save_multiple_congestion_datasets(num_datasets=5, num_points=200, max_latency=60, noise_level=0.2):
    # Get the path of the current script file and create output folders
    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(script_dir, "Datasets")
    plots_dir = os.path.join(script_dir, "Plots")
    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Generate congestion datasets
    for i in range(num_datasets):
        # Make discernible knees slightly more common than before.
        clear_knee = np.random.rand() > 0.5  # 1/2 probability for a clear knee, 1/2 for no knee
        
        # Generate the dataset with varying knee speeds for clear knee datasets
        if clear_knee:
            knee_speed = np.random.uniform(20, 50)  # Random knee speed between 20 and 50 Mbps
            speeds, latency = generate_congestion_data(num_points=num_points, max_latency=max_latency, noise_level=noise_level, knee_speed=knee_speed, clear_knee=clear_knee)
            plot_title = f"Dataset {i+1} (with knee at {knee_speed:.0f} Mbps)"
        else:
            speeds, latency = generate_congestion_data(num_points=num_points, max_latency=max_latency, noise_level=noise_level, knee_speed=0, clear_knee=clear_knee)
            plot_title = f"Dataset {i+1} (no discernible knee)"
        
        # Plot and save the dataset
        plot_and_save_dataset(speeds, latency, plot_title, i+1, datasets_dir, plots_dir)

    # Generate two more proportional trend datasets (linear with a not very steep gradient)
    for j in range(2):
        slope = np.random.uniform(0.1, 0.25)  # Random slope for gentle gradient
        intercept = np.random.uniform(5, 10)  # Random intercept between 5 and 10 ms
        speeds, latency = generate_proportional_trend(num_points=num_points, slope=slope, intercept=intercept, max_latency_diff=15, noise_level=noise_level)
        plot_title = f"Proportional Trend {j+1} (slope={slope:.2f}, intercept={intercept:.2f})"
        
        # Plot and save the proportional trend dataset
        plot_and_save_dataset(speeds, latency, plot_title, num_datasets + j + 1, datasets_dir, plots_dir)

def main():
    plot_and_save_multiple_congestion_datasets(
        num_datasets=5,
        num_points=200,
        max_latency=60,
        noise_level=0.2,
    )


if __name__ == "__main__":
    main()
