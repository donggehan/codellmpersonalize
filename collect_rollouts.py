import subprocess
import yaml
import time
import sys

def run_command(command):
    """Run the given command and retry if it fails."""
    max_attempts = 2
    for attempt in range(max_attempts):
        try:
            subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            print(f"Command executed successfully: {command}")
            return
        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt+1}/{max_attempts} failed: {e.output}")
            if attempt < max_attempts - 1:
                time.sleep(5)  # wait for 5 seconds before retrying
            else:
                print(f"Command failed after {max_attempts} attempts: {command}")

def main():
    # Load variables from YAML file
    with open('cos_eor/configs/local/envs.yaml', 'r') as file:
        vars_list = file.read().splitlines()

    # Loop through each variable and run commands
    for var in vars_list:
        for i in range(5):
            print (f'running scene {var}, iteration {i}')
            run_command(f"./run_cli.sh test {var} 10")

if __name__ == "__main__":
    main()

