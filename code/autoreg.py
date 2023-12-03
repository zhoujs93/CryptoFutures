import os
import subprocess

if __name__ == '__main__':
    btc_command = 'btcli subnet register --wallet.name miner_live --wallet.hotkey default'
    command = f'conda activate python310 && {btc_command}'
    subprocess.run(["conda activate python310", ""], capture_output = True, shell = True)