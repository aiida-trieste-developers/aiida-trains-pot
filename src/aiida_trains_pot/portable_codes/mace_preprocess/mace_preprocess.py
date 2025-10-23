# ruff: noqa
import os
import subprocess

import yaml


def load_config(file_path):
    with open(file_path) as file:
        return yaml.safe_load(file)


preprocess_path = os.environ.get("preprocess_path")


def build_command(params):
    command = [
        "mace_prepare_data",
        "--train_file",
        "training.xyz",
        "--valid_file",
        "validation.xyz",
        "--test_file",
        "test.xyz",
        "--energy_key",
        "dft_energy",
        "--forces_key",
        "dft_forces",
        "--stress_key",
        "dft_stress",
        "--r_max",
        str(params["r_max"]),
        "--compute_statistics",
        "",
        "--h5_prefix",
        "processed_data/",
        "--seed",
        str(params["seed"]),
    ]
    return [arg for arg in command if arg]


def run_command(command):
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}.")
        print(f"Error output:\n{e.stderr}")
        raise


def main():
    print("Start processing")
    config_path = "config.yml"
    params = load_config(config_path)
    print("Loaded parameters:", params)

    command = build_command(params)
    print("Running command:", " ".join(command))

    result = run_command(command)
    print("Command finished with exit status:", result.returncode)
    print("Finished processing")


if __name__ == "__main__":
    main()
