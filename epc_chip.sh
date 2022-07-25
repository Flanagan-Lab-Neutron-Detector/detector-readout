#!/bin/bash

# Arguments:
#   1    Serial port
#   2    directory to save to

port=$1
savedir=$2

mv_start=5000
mv_stop=7000
mv_step=100

echo python3 Neutron_CLI.py --p $port -e --all-sectors
python3 Neutron_CLI.py --p $port -e --all-sectors

echo sleep 10m # delay to allow chip to erase
sleep 10m

echo python3 Neutron_CLI.py --p $port -w -v 0 --all-sectors
python3 Neutron_CLI.py --p $port -w -v 0 --all-sectors

echo sleep 1h # delay to allow chip to write
sleep 1h

echo python3 Neutron_CLI.py --p $port -r --bitcount --all-sectors --start $mv_start --stop $mv_stop --step $mv_step -d $savedir
python3 Neutron_CLI.py --p $port -r --bitcount --all-sectors --start $mv_start --stop $mv_stop --step $mv_step -d $savedir

echo 'Complete!'
