import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import MinMaxScaler

DEFAULT_MIN_KNEE_SPEED_RATIO = 0.6


def detect_knee_with_gpr(
    speeds,
    latency,
    curvature_threshold=0.3,
    similarity_threshold=0.1,
    min_knee_speed_ratio=DEFAULT_MIN_KNEE_SPEED_RATIO,
    edge_exclusion_ratio=0.1,
):
    scaler = MinMaxScaler()
    speeds_scaled = scaler.fit_transform(speeds[:, np.newaxis]).ravel()

    kernel = C(1.0, (1e-2, 1e3)) * RBF(length_scale=0.2, length_scale_bounds=(1e-2, 10.0)) + WhiteKernel(
        noise_level=1.0,
        noise_level_bounds=(1e-5, 1e2),
    )
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-4,
        normalize_y=True,
        n_restarts_optimizer=10,
        random_state=42,
    )
    gpr.fit(speeds_scaled[:, np.newaxis], latency)

    speeds_pred_scaled = np.linspace(0, 1, 1000)
    latency_pred, sigma = gpr.predict(speeds_pred_scaled[:, np.newaxis], return_std=True)
    speeds_pred = scaler.inverse_transform(speeds_pred_scaled[:, np.newaxis]).ravel()

    slope = np.gradient(latency_pred, speeds_pred)
    curvature = np.gradient(slope, speeds_pred)

    avg_low = np.mean(latency[:20])
    avg_high = np.mean(latency[-20:])
    relative_change = np.abs(avg_high - avg_low) / max(avg_low, 1e-10)

    result = {
        "status": "no_knee",
        "message": "",
        "knee_speed": None,
        "knee_index": None,
        "avg_low": avg_low,
        "avg_high": avg_high,
        "relative_change": relative_change,
        "speeds_pred": speeds_pred,
        "latency_pred": latency_pred,
        "sigma": sigma,
        "slope": slope,
        "curvature": curvature,
        "curvature_threshold": curvature_threshold,
        "similarity_threshold": similarity_threshold,
        "min_knee_speed_ratio": min_knee_speed_ratio,
    }

    if relative_change <= similarity_threshold:
        result["message"] = "No discernible knee: low-speed and high-speed latency are too similar."
        print(result["message"])
        return result

    edge_margin = max(1, int(len(speeds_pred) * edge_exclusion_ratio))
    candidate_slice = slice(edge_margin, len(speeds_pred) - edge_margin)
    interior_curvature = curvature[candidate_slice]

    if interior_curvature.size == 0:
        result["message"] = "No knee detected: insufficient interior region after edge exclusion."
        print(result["message"])
        return result

    local_index = int(np.argmax(interior_curvature))
    knee_index = local_index + edge_margin
    knee_speed = speeds_pred[knee_index]
    peak_curvature = curvature[knee_index]

    result["knee_index"] = knee_index
    result["knee_speed"] = knee_speed

    if peak_curvature < curvature_threshold:
        result["message"] = (
            f"No knee detected: curvature peak {peak_curvature:.2f} is below threshold "
            f"{curvature_threshold:.2f}."
        )
        print(result["message"])
        return result

    min_knee_speed = np.max(speeds) * min_knee_speed_ratio
    knee_below_min_speed = knee_speed < min_knee_speed

    post_knee_mask = speeds_pred >= knee_speed
    pre_knee_mask = speeds_pred < knee_speed

    if np.count_nonzero(pre_knee_mask) < 5 or np.count_nonzero(post_knee_mask) < 5:
        result["message"] = "No knee detected: not enough points before or after the candidate knee."
        print(result["message"])
        return result

    pre_knee_slope = np.mean(slope[pre_knee_mask][-50:])
    post_knee_slope = np.mean(slope[post_knee_mask][:50])

    if post_knee_slope <= pre_knee_slope * 1.5:
        result["message"] = (
            f"No knee detected: post-knee slope {post_knee_slope:.2f} does not increase enough over "
            f"pre-knee slope {pre_knee_slope:.2f}."
        )
        print(result["message"])
        return result

    result["status"] = "critical_knee"
    if knee_below_min_speed:
        result["message"] = (
            f"Critical knee detected at speed: {knee_speed:.2f} Mbps "
            f"(below preferred minimum {min_knee_speed:.2f} Mbps, curvature peak {peak_curvature:.2f}, "
            f"post-knee slope {post_knee_slope:.2f})."
        )
    else:
        result["message"] = (
            f"Critical knee detected at speed: {knee_speed:.2f} Mbps "
            f"(curvature peak {peak_curvature:.2f}, post-knee slope {post_knee_slope:.2f})."
        )
    print(result["message"])
    return result


def visualize_dataset(dataset_file, speeds, latency, detection_result, output_dir):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=False)

    title_color = "forestgreen" if detection_result["status"] == "critical_knee" else "firebrick"

    axes[0].scatter(speeds, latency, color="firebrick", s=20, alpha=0.7, label="Observed data")
    axes[0].plot(
        detection_result["speeds_pred"],
        detection_result["latency_pred"],
        color="navy",
        linewidth=2,
        label="GPR mean",
    )
    axes[0].fill_between(
        detection_result["speeds_pred"],
        detection_result["latency_pred"] - 1.96 * detection_result["sigma"],
        detection_result["latency_pred"] + 1.96 * detection_result["sigma"],
        color="skyblue",
        alpha=0.3,
        label="95% confidence band",
    )

    if detection_result["knee_speed"] is not None:
        knee_speed = detection_result["knee_speed"]
        knee_latency = np.interp(
            knee_speed,
            detection_result["speeds_pred"],
            detection_result["latency_pred"],
        )
        axes[0].axvline(knee_speed, color=title_color, linestyle=":", linewidth=2)
        axes[0].scatter([knee_speed], [knee_latency], color=title_color, s=60, zorder=5)

    axes[0].set_title(
        f"{dataset_file}: {'Knee detected' if detection_result['status'] == 'critical_knee' else 'No knee detected'}",
        color=title_color,
    )
    axes[0].set_ylabel("Latency (ms)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")
    axes[0].text(
        0.02,
        0.04,
        detection_result["message"],
        transform=axes[0].transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": title_color},
    )

    axes[1].plot(
        detection_result["speeds_pred"],
        detection_result["slope"],
        color="darkorange",
        linewidth=1.8,
        label="Slope",
    )
    if detection_result["knee_speed"] is not None:
        axes[1].axvline(detection_result["knee_speed"], color=title_color, linestyle=":", linewidth=2)
    axes[1].set_ylabel("dLatency / dSpeed")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left")

    axes[2].plot(
        detection_result["speeds_pred"],
        detection_result["curvature"],
        color="purple",
        linewidth=1.8,
        label="Curvature",
    )
    axes[2].axhline(
        detection_result["curvature_threshold"],
        color="purple",
        linestyle="--",
        linewidth=1.3,
        label="Curvature threshold",
    )
    if detection_result["knee_speed"] is not None:
        axes[2].axvline(detection_result["knee_speed"], color=title_color, linestyle=":", linewidth=2)
    axes[2].set_xlabel("Speed (Mbps)")
    axes[2].set_ylabel("d2Latency / dSpeed2")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper left")

    fig.tight_layout()

    output_file = os.path.join(output_dir, f"{os.path.splitext(dataset_file)[0]}_gpr.png")
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {output_file}")
    plt.close(fig)


def load_and_process_datasets_with_gpr():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(script_dir, "Datasets")
    plots_dir = os.path.join(script_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    dataset_files = sorted(f for f in os.listdir(datasets_dir) if f.startswith("dataset_") and f.endswith(".txt")) if os.path.isdir(datasets_dir) else []

    if not dataset_files:
        print("No dataset files found.")
        return

    for dataset_file in dataset_files:
        file_path = os.path.join(datasets_dir, dataset_file)
        print(f"\nProcessing {dataset_file}...")

        data = np.loadtxt(file_path, skiprows=1)
        speeds = data[:, 0]
        latency = data[:, 1]

        detection_result = detect_knee_with_gpr(speeds, latency)
        visualize_dataset(dataset_file, speeds, latency, detection_result, plots_dir)


def main():
    load_and_process_datasets_with_gpr()


if __name__ == "__main__":
    main()
