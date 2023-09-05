param(
	[string]$port="COM4",
	[int]$start=4000,
	[int]$stop=7000,
	[int]$step=100,

	[Parameter(Mandatory=$true)]
	[string]$dir
)

Write-Host "Erase all sectors"
python Neutron_CLI.py -p $port erase-chip

Write-Host "Erase delay 10m"
Start-Sleep -s 600

Write-Host "Program device"
python Neutron_CLI.py -p $port write-chip -v 0

Write-Host "Program delay 1h"
Start-Sleep -s 3600

Write-Host "Read all sectors"
python Neutron_CLI.py -p $port read --address 0 --sectors 1024 --start $start --stop $stop --step $step -d $dir --format=binary

Write-Host "Complete"
