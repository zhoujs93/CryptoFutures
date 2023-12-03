#!/bin/bash
if [ $(/bin/pgrep -f "miner_nbeats.py") ]; then
    echo "script running"
else
    echo "script not running"
    tmux new-session -d; send-keys "source activate python310 && cd /home/putsncalls23/CryptoFutures/time-series-prediction-subnet/neurons && python miner_nbeats.py --netuid 8 --wallet.name miner_live --wallet.hotkey default --logging.debug --axon.port 9002" Enter
fi
