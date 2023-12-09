import subprocess
import time

from google.cloud import storage

def upload_to_gcs(bucket_name, local_file_path, destination_blob_name):
    # Initialize a client using the environment variable GOOGLE_APPLICATION_CREDENTIALS
    client = storage.Client()

    # Get the bucket
    bucket = client.get_bucket(bucket_name)

    # Create a blob object
    blob = bucket.blob(destination_blob_name)

    # Upload the local file
    blob.upload_from_filename(local_file_path)

    print(f"File {local_file_path} uploaded to {bucket_name}/{destination_blob_name}")


import re


def format_journal_output(raw_output):
    # Split the raw output into individual log entries
    log_entries = re.split(r'(?<=\d{2}:\d{2}:\d{2})\s', raw_output)

    # Remove any empty entries
    log_entries = [entry.strip() for entry in log_entries if entry.strip()]

    # Format each log entry for better readability
    formatted_entries = []
    for entry in log_entries:
        timestamp, rest_of_entry = entry.split(' python')
        formatted_entry = f"{timestamp} | {rest_of_entry}"
        formatted_entries.append(formatted_entry)

    # Join the formatted entries into a single string
    # formatted_output = '\n'.join(formatted_entries)

    return formatted_entries


def export_and_append_logs(unit_name, num_lines=10, existing_log_file='existing_logs.txt'):
    # Run journalctl command to retrieve logs for the specified unit
    try:
        # Build the journalctl command with specified options
        command = ["journalctl", "-n", str(num_lines), "-u", unit_name]

        # Run the command and capture the output
        log_entries = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

        log_entries = ' '.join(log_entries.stdout.split('Dec '))
        # formatted_output = format_journal_output(result)
        #

        # formatted_log = log_entries
        existing_logs = ''
        try:
            with open(existing_log_file, 'r') as file:
                existing_logs = file.readlines()
        except FileNotFoundError:
            pass

        # Append newly retrieved logs to existing logs
        output = existing_logs + log_entries
        all_logs = output.split('\n')
        if len(all_logs) > 5000:
            all_logs = all_logs[-3000:]
        #
        # all_logs = list(reversed(all_logs))
        # output = '\n'.join(all_logs)

        with open(existing_log_file, 'w') as file:
            file.write(output)

        print(f"Journalctl output written to {existing_log_file}")

    except subprocess.CalledProcessError as e:
        # Handle the case where the command returns a non-zero exit status
        print(f"Error running journalctl command: {e}")
        print(f"Error output: {e.stderr}")

    all_logs = list(reversed(all_logs))
    # Write combined logs back to the file
    with open(existing_log_file, 'w') as file:
        file.writelines(all_logs)


if __name__ == "__main__":
    # Specify the systemd unit name for which you want to export logs
    while True:
        unit_name = "miner_nbeats"

        # Specify the number of entries to retrieve from journalctl
        num_entries = 3000

        # Specify the path to the existing log file
        filename = 'miner_live_default_9001.txt'
        existing_log_file = f'./logs/{filename}'

        export_and_append_logs(unit_name, num_entries, existing_log_file)
        # gcloud auth application-default login
        bucket_name = 'tao-mining-logs'
        destination_blob_name = f'putsncalls/{filename}'
        upload_to_gcs(bucket_name, existing_log_file, destination_blob_name)

        time.sleep(3000)