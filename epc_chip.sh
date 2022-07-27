#!/bin/bash

# Arguments:
#   1    Serial port
#   2    directory to save to

port=$1
savedir=$2

mv_start=5000
mv_stop=7000
mv_step=100

log () {
	echo "[`date '+%Y-%m-%d %H:%M:%S'`] $1"
}

log "Erase all sectors"
python3 Neutron_CLI.py --p $port -e --all-sectors

log "Erase delay 10m"
sleep 10m # delay to allow the chip to erase

log "Program device"
python3 Neutron_CLI.py --p $port -w -v 0 --all-sectors

log "Program delay 1h"
sleep 1h # delay to allow the device to write and then settle

log "Writing to $savedir"
log "Bit count all sectors in mV range [$mv_start, $mv_stop) by $ms_step mV"
python3 Neutron_CLI.py --p $port -r --bitcount --all-sectors --start $mv_start --stop $mv_stop --step $mv_step -d $savedir

log "Complete!"
