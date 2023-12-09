#!/bin/bash
PATH=/opt/conda/bin:/opt/conda/condabin:/usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games

WALLET_NAME="default6"
PORT_NUMBER="9006"

# ExecStart=/opt/conda/bin/python miner_nbeats.py --netuid 8 --wallet.name miner_live --wallet.hotkey default4 --logging.debug --logging.trace --axon.port 9004

[Unit]
Description=Mining service for nbeats
After=network.target

[Service]
Type=simple
User=putsncalls23
WorkingDirectory=/home/putsncalls23/CryptoFutures/time-series-prediction-subnet/neurons
ExecStart=/opt/conda/envs/python310/bin/python miner_lstm.py --netuid 8 --wallet.name miner_live --wallet.hotkey default7 --logging.debug --axon.port 9007 --base_model model_v4_1
Restart=always
RestartSec=30

[Install]
WantedBy=default.target

# steps to start miner: 1) start google cloud
# 2) set up port
# 3) set up systemd
# sudo nano /etc/systemd/system/miner_nbeats.service
# sudo systemctl daemon-reload
# sudo systemctl start miner_nbeats
# sudo systemctl enable miner_nbeats
# sudo systemctl status miner_nbeats
# sudo systemctl stop miner_nbeats
# sudo systemctl restart miner_nbeats
# journalctl -n 1000 -u miner_nbeats