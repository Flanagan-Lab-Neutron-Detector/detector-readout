#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Arguments:
#   1    Serial port
#   2    directory to save to

port=${1:-}
if [[ -z "$port" ]]; then
	echo "usage: $0 port savedir mvstart mvstop mvstep"
	exit 1
fi
savedir=${2:-}
if [[ -z "$savedir" ]]; then
	echo "usage: $0 port savedir mvstart mvstop mvstep"
	exit 1
fi

mv_start=${3:-4000}
mv_stop=${4:-7000}
mv_step=${5:-100}

# TODO: Validate parameters

log () {
	echo "[`date '+%Y-%m-%d %H:%M:%S'`] $1"
}

log "Starting E/P/R"
log "Port:    $port"
log "Savedir: $savedir"
log "Start:   $mv_start mV"
log "Stop:    $mv_stop mV"
log "Step:    $mv_step mV"

log "Erase all sectors"
python3 Neutron_CLI.py --p $port -e --all-sectors

log "Erase delay 10m"
sleep 10m # delay to allow the chip to erase

log "Program device"
python3 Neutron_CLI.py --p $port -w -v 0 --all-sectors

log "Program delay 1h"
sleep 1h # delay to allow the device to write and then settle

log "Writing to $savedir"
log "Read all sectors in range [$mv_start, $mv_stop) mV in steps of $mv_step mV"
python3 Neutron_CLI.py --p $port -r --all-sectors --start $mv_start --stop $mv_stop --step $mv_step -d $savedir

log "Complete!"
