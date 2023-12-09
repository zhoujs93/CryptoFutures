import subprocess
import re
import time

def run_btcli_command(command, input_data=None):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, universal_newlines=True)
    output, _ = process.communicate(input=input_data)
    return output

def parse_balance(output):
    balance_match = re.search(r'Your balance is: ([\d.]+)', output)
    if balance_match:
        return float(balance_match.group(1))
    return None


if __name__ == "__main__":

    while True:
        btcli_command = [
            'btcli', 'subnet', 'register',
            '--netuid', '8',
            '--wallet.name', 'miner_live',
            '--wallet.hotkey', 'default3',
            '--subtensor.network', 'local'
        ]

        output = run_btcli_command(btcli_command)
        print(output)

        time.sleep(15)

        balance_match = re.search(r'Your balance is: ([\d.]+)', output)
        if balance_match:
            bal = float(balance_match.group(1))
            print(f'Result matched balance is {bal}')

        tau = 'τ'
        cost_prompt = output.split(tau)
        balance = cost_prompt[-1].split('\n')[0]
        print(f'Cost to register is {float(balance)}')
        if float(balance) < 4.4:
            print(f'Proceeding with registration')

            # Automatically send 'y' to continue

            continue_input = 'y\n'
            run_btcli_command(btcli_command, input_data=continue_input)
            time.sleep(30)

            password = "@Jz19930525"
            run_btcli_command(btcli_command, input_data = password)

        else:
            continue_input = 'n\n'
            run_btcli_command(btcli_command, input_data=continue_input)

        time.sleep(30)