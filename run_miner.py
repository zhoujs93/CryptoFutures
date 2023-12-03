import psutil
import subprocess
import os
import time

def is_process_running(process_name):
    for process in psutil.process_iter(['pid', 'name']):
        if process.info['name'] == process_name:
            return True
    return False

def main():
    anaconda_env = "python310"
    python_script = "miner_nbeats.py"

    if is_process_running("python") and is_process_running(python_script):
        print(f"{python_script} is already running.")
    else:
        # activate_cmd = f"conda activate {anaconda_env} && cd CryptoFutures/time-series-predictiono-subnet/neurons/ " \
        #                f"&& python {python_script} --netuid 8 --wallet.name miner_live --wallet.hotkey default --logging.debug" \
        #                f" --axon.port 9002"

        # Run the command in a tmux session
        print(f'{python_script} not running')
        tmux_cmd = f"bash run_script.sh"
        subprocess.run(tmux_cmd, shell=True)

        print(f"Started {python_script} in a new tmux session.")

if __name__ == '__main__':
    while True:
        main()
        time.sleep(300)
