import numpy as np
import os 
import matplotlib.pyplot as plt

DEFAULT_MIN_KNEE_SPEED_RATIO = 0.6

# Function to calculate the exponentially weighted average (EWA)
def calculate_ewa(data, smoothing_factor=0.3):
    ewa = np.zeros(len(data))
    ewa[0] = data[0]  # Initialize the first value
    for i in range(1, len(data)):
        ewa[i] = smoothing_factor * data[i] + (1 - smoothing_factor) * ewa[i - 1]
    return ewa

# Function to calculate running average noise
def calculate_noise(latency, ewa):
    # Noise is the difference between the actual latency and the EWA-smoothed latency
    noise = np.abs(latency - ewa)
    return noise


def calculate_noise_change(noise):
    noise_change = np.full_like(noise, np.nan, dtype=float)
    if len(noise) > 1:
        noise_change[1:] = noise[1:] - noise[:-1]
    return noise_change

# Function to detect a knee and determine whether to limit speed or not, considering both latency and noise
def detect_knee(speeds, latency, num_compare_points=20, smoothing_factor=0.3, change_threshold=5, similarity_threshold=0.1, noise_increase_threshold=0.5, min_knee_speed_ratio=DEFAULT_MIN_KNEE_SPEED_RATIO, steep_gradient_threshold=5):
    # Compare the first and last 20 points
    lower_speed_latency = latency[:num_compare_points]
    higher_speed_latency = latency[-num_compare_points:]
    
    # Calculate the average latency for the lower and higher speeds
    avg_low = np.mean(lower_speed_latency)
    avg_high = np.mean(higher_speed_latency)
    
    # Compare the average latencies (check if they are within 10% similarity)
    ewa_low_to_high = calculate_ewa(latency, smoothing_factor)
    ewa_high_to_low = calculate_ewa(latency[::-1], smoothing_factor)[::-1]  # Reverse to calculate from the other side
    ewa_diff = np.abs(ewa_low_to_high - ewa_high_to_low)
    noise = calculate_noise(latency, ewa_low_to_high)
    noise_increase = calculate_noise_change(noise)
    gradient = np.gradient(latency)

    result = {
        "knee_speed": None,
        "knee_index": None,
        "message": "",
        "status": "no_knee",
        "avg_low": avg_low,
        "avg_high": avg_high,
        "ewa_low_to_high": ewa_low_to_high,
        "ewa_high_to_low": ewa_high_to_low,
        "ewa_diff": ewa_diff,
        "noise": noise,
        "noise_increase": noise_increase,
        "gradient": gradient,
        "change_threshold": change_threshold,
        "noise_increase_threshold": noise_increase_threshold,
        "steep_gradient_threshold": steep_gradient_threshold,
    }

    if np.abs(avg_high - avg_low) / max(avg_low, 1e-10) <= similarity_threshold:
        result["message"] = "No discernible knee: latencies at lower and higher speeds are similar."
        print(result["message"])
        return result  # No knee detected
    
    # Find the first point where the difference exceeds the threshold
    knee_index = np.where(ewa_diff > change_threshold)[0]
    
    if len(knee_index) == 0:
        result["message"] = "No knee detected: changes in latency are too small."
        print(result["message"])
        return result  # No knee detected
    
    # Determine the speed at which the knee happens
    first_knee_index = int(knee_index[0])
    knee_speed = speeds[first_knee_index]
    result["knee_index"] = first_knee_index
    result["knee_speed"] = knee_speed

    # Check if the knee speed is at least min_knee_speed_ratio * 100% of the maximum observed speed
    max_speed = np.max(speeds)
    min_knee_speed = max_speed * min_knee_speed_ratio

    if knee_speed < min_knee_speed:
        result["message"] = (
            f"Knee detected at {knee_speed:.2f} Mbps, but it's below {min_knee_speed_ratio * 100:.0f}% of the maximum speed "
            f"({min_knee_speed:.2f} Mbps). Ignoring."
        )
        print(result["message"])
        return result

    # Ensure the knee corresponds to a steep gradient (change in latency is sharp)
    if np.max(np.abs(gradient)) < steep_gradient_threshold:
        result["message"] = f"Knee detected at {knee_speed:.2f} Mbps, but it's not steep enough. Ignoring."
        print(result["message"])
        return result

    # Detect a sudden increase in noise (50% increase or more)
    significant_noise_index = np.where(np.nan_to_num(noise_increase, nan=-np.inf) > noise_increase_threshold)[0]

    if len(significant_noise_index) == 0:
        result["message"] = f"Knee detected at {knee_speed:.2f} Mbps, but no significant noise increase. Ignoring."
        print(result["message"])
        return result

    result["status"] = "critical_knee"
    result["message"] = (
        f"Critical knee detected at speed: {knee_speed:.2f} Mbps "
        f"(at least {min_knee_speed_ratio * 100:.0f}% of max speed, steep gradient, and sudden noise increase)."
    )
    print(result["message"])
    return result


def visualize_dataset(dataset_file, speeds, latency, detection_result, output_dir):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(speeds, latency, color="steelblue", linewidth=1.5, label="Latency")
    axes[0].plot(
        speeds,
        detection_result["ewa_low_to_high"],
        color="darkorange",
        linewidth=2,
        label="EWA (low to high)",
    )
    axes[0].plot(
        speeds,
        detection_result["ewa_high_to_low"],
        color="seagreen",
        linewidth=2,
        linestyle="--",
        label="EWA (high to low)",
    )

    title_color = "forestgreen" if detection_result["status"] == "critical_knee" else "firebrick"
    axes[0].set_title(f"{dataset_file}: {'Knee detected' if detection_result['status'] == 'critical_knee' else 'No knee detected'}", color=title_color)
    axes[0].set_ylabel("Latency")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    if detection_result["knee_index"] is not None:
        knee_speed = detection_result["knee_speed"]
        knee_latency = latency[detection_result["knee_index"]]
        knee_color = "forestgreen" if detection_result["status"] == "critical_knee" else "firebrick"
        axes[0].axvline(knee_speed, color=knee_color, linestyle=":", linewidth=2)
        axes[0].scatter([knee_speed], [knee_latency], color=knee_color, s=60, zorder=5)
        axes[0].annotate(
            f"{knee_speed:.2f} Mbps",
            xy=(knee_speed, knee_latency),
            xytext=(10, 10),
            textcoords="offset points",
            color=knee_color,
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.85, "edgecolor": knee_color},
        )

    axes[0].text(
        0.02,
        0.04,
        detection_result["message"],
        transform=axes[0].transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": title_color},
    )

    axes[1].plot(speeds, detection_result["ewa_diff"], color="purple", linewidth=1.8, label="EWA difference")
    axes[1].axhline(
        detection_result["change_threshold"],
        color="purple",
        linestyle=":",
        linewidth=1.5,
        label="EWA diff threshold",
    )

    if detection_result["knee_index"] is not None:
        axes[1].axvline(detection_result["knee_speed"], color=title_color, linestyle=":", linewidth=2)

    axes[1].set_ylabel("EWA difference")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left")

    axes[2].plot(speeds, detection_result["noise_increase"], color="teal", linewidth=1.4, label="Noise change")
    axes[2].axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label="Zero change",
    )
    axes[2].axhline(
        detection_result["noise_increase_threshold"],
        color="teal",
        linestyle=":",
        linewidth=1.5,
        label="Noise change threshold",
    )
    axes[2].axhline(
        -detection_result["noise_increase_threshold"],
        color="teal",
        linestyle=":",
        linewidth=1.0,
        alpha=0.5,
    )

    if detection_result["knee_index"] is not None:
        axes[2].axvline(detection_result["knee_speed"], color=title_color, linestyle=":", linewidth=2)

    axes[2].set_xlabel("Speed (Mbps)")
    axes[2].set_ylabel("Noise change")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper left")

    fig.tight_layout()

    output_file = os.path.join(output_dir, f"{os.path.splitext(dataset_file)[0]}_standard.png")
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {output_file}")
    plt.close(fig)

# Function to load datasets from files and apply EWA-based knee detection with noise consideration
def load_and_process_datasets():
    # Get the path of the current script file
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the Python script
    datasets_dir = os.path.join(script_dir, "Datasets")
    plots_dir = os.path.join(script_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Find the text files generated by the fake data script
    dataset_files = sorted(f for f in os.listdir(datasets_dir) if f.startswith("dataset_") and f.endswith(".txt")) if os.path.isdir(datasets_dir) else []
    
    if not dataset_files:
        print("No dataset files found.")
        return

    # Process each dataset file
    for dataset_file in dataset_files:
        file_path = os.path.join(datasets_dir, dataset_file)
        print(f"\nProcessing {dataset_file}...")

        # Load the dataset (speeds and latency) from the ASCII file
        data = np.loadtxt(file_path, skiprows=1)  # Skip the header row
        speeds = data[:, 0]
        latency = data[:, 1]

        # Detect the knee for this dataset using EWA and noise detection
        detection_result = detect_knee(speeds, latency)
        visualize_dataset(dataset_file, speeds, latency, detection_result, plots_dir)

# Main function
def main():
    load_and_process_datasets()

if __name__ == "__main__":
    main()
