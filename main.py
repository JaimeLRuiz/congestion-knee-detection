import argparse
import os
import shutil
import sys
from datetime import datetime

import CongestionDataGen
import KneeDetectionGaussian
import KneeDetectionStandard


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "Datasets")
PLOTS_DIR = os.path.join(BASE_DIR, "Plots")
EXPORTS_DIR = os.path.join(BASE_DIR, "Exports")


def ensure_directories():
    os.makedirs(DATASETS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def clear_directory_contents(directory):
    if not os.path.isdir(directory):
        return

    for entry in os.listdir(directory):
        path = os.path.join(directory, entry)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def clear_outputs():
    ensure_directories()
    clear_directory_contents(DATASETS_DIR)
    clear_directory_contents(PLOTS_DIR)
    print(f"Cleared {DATASETS_DIR}")
    print(f"Cleared {PLOTS_DIR}")


def run_actions(sequence, clear_first=False, num_datasets=5, num_points=200, max_latency=60, noise_level=0.2):
    ensure_directories()

    if clear_first:
        clear_outputs()

    for action in sequence:
        if action == "generate":
            print("Running congestion data generation...")
            CongestionDataGen.plot_and_save_multiple_congestion_datasets(
                num_datasets=num_datasets,
                num_points=num_points,
                max_latency=max_latency,
                noise_level=noise_level,
            )
        elif action == "standard":
            print("Running standard knee detection...")
            KneeDetectionStandard.load_and_process_datasets()
        elif action == "gaussian":
            print("Running Gaussian knee detection...")
            KneeDetectionGaussian.load_and_process_datasets_with_gpr()


def export_outputs(export_name=None):
    ensure_directories()
    os.makedirs(EXPORTS_DIR, exist_ok=True)

    if export_name:
        folder_name = export_name
    else:
        folder_name = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    export_dir = os.path.join(EXPORTS_DIR, folder_name)
    datasets_export_dir = os.path.join(export_dir, "Datasets")
    plots_export_dir = os.path.join(export_dir, "Plots")

    if os.path.exists(export_dir):
        raise FileExistsError(f"Export folder already exists: {export_dir}")

    shutil.copytree(DATASETS_DIR, datasets_export_dir)
    shutil.copytree(PLOTS_DIR, plots_export_dir)

    print(f"Created export folder: {export_dir}")
    return export_dir


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run congestion dataset generation and knee detection workflows."
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a sequence of workflow steps.")
    run_parser.add_argument(
        "--sequence",
        nargs="+",
        choices=["generate", "standard", "gaussian"],
        required=True,
        help="Ordered steps to execute.",
    )
    run_parser.add_argument(
        "--clear-first",
        action="store_true",
        help="Clear Datasets and Plots before running the sequence.",
    )
    run_parser.add_argument("--num-datasets", type=int, default=5, help="Number of congestion datasets to generate.")
    run_parser.add_argument("--num-points", type=int, default=200, help="Number of points per dataset.")
    run_parser.add_argument("--max-latency", type=float, default=60, help="Maximum simulated latency.")
    run_parser.add_argument("--noise-level", type=float, default=0.2, help="Noise level used in data generation.")

    clear_parser = subparsers.add_parser("clear", help="Clear Datasets and Plots.")
    clear_parser.add_argument(
        "--yes",
        action="store_true",
        help="Proceed without an interactive confirmation prompt.",
    )

    export_parser = subparsers.add_parser("export", help="Duplicate Datasets and Plots into an export folder.")
    export_parser.add_argument(
        "--name",
        help="Optional name for the export folder under Exports.",
    )

    return parser


def ask_yes_no(prompt, default=False):
    suffix = "[Y/n]" if default else "[y/N]"
    reply = input(f"{prompt} {suffix}: ").strip().lower()
    if not reply:
        return default
    return reply in {"y", "yes"}


def run_menu():
    print("Choose a run sequence:")
    print("1. Run generate -> standard")
    print("2. Run generate -> gaussian")
    print("3. Run generate -> standard -> gaussian")
    print("4. Exit")

    choice = input("Enter choice [1-4]: ").strip()

    if choice == "1":
        run_actions(sequence=["generate", "standard"])
        return True
    if choice == "2":
        run_actions(sequence=["generate", "gaussian"])
        return True
    if choice == "3":
        run_actions(sequence=["generate", "standard", "gaussian"])
        return True

    print("Exit.")
    return False


def prompt_for_command():
    while True:
        print("No command provided.")
        print("Select an action:")
        print("1. Run generate -> standard")
        print("2. Run generate -> gaussian")
        print("3. Run generate -> standard -> gaussian")
        print("4. Clear Datasets and Plots")
        print("5. Export Datasets and Plots")
        print("6. Exit")

        choice = input("Enter choice [1-6]: ").strip()

        if choice == "1":
            run_actions(sequence=["generate", "standard"])
            return
        if choice == "2":
            run_actions(sequence=["generate", "gaussian"])
            return
        if choice == "3":
            run_actions(sequence=["generate", "standard", "gaussian"])
            return
        if choice == "4":
            if ask_yes_no("Clear Datasets and Plots?", default=False):
                clear_outputs()
            else:
                print("Clear cancelled.")
            if not run_menu():
                return
            continue
        if choice == "5":
            export_name = input("Optional export folder name (leave blank for timestamp): ").strip()
            export_outputs(export_name=export_name or None)
            if ask_yes_no("Clear Datasets and Plots after exporting?", default=False):
                clear_outputs()
            if not run_menu():
                return
            continue

        print("Exit.")
        return


def main():
    parser = build_parser()
    if len(sys.argv) == 1:
        prompt_for_command()
        return

    args = parser.parse_args()

    if args.command == "run":
        run_actions(
            sequence=args.sequence,
            clear_first=args.clear_first,
            num_datasets=args.num_datasets,
            num_points=args.num_points,
            max_latency=args.max_latency,
            noise_level=args.noise_level,
        )
    elif args.command == "clear":
        if not args.yes:
            confirmation = input("Clear Datasets and Plots? [y/N]: ").strip().lower()
            if confirmation not in {"y", "yes"}:
                print("Clear cancelled.")
                return
        clear_outputs()
    elif args.command == "export":
        export_outputs(export_name=args.name)


if __name__ == "__main__":
    main()
