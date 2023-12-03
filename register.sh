#!/usr/bin/expect

set timeout -1

set hotkey [lindex $argv 0]

while {1} {
        spawn btcli subnet register --wallet.name miner_live --wallet.hotkey default --subtensor.network finney --netuid 8

        expect "Do you want to continue*"
        send "y\r"

        expect "Enter password to unlock key*"
        send "@Jz19930525\r"

        expect "Recycle*"
        send "y\r"

        interact
}