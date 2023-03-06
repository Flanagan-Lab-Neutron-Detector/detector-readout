param(
	[string]$port="COM4",
	[int]$start=4000,
	[int]$stop=7000,
	[int]$step=100,
	
	[Parameter(Mandatory=$true)]
	[string]$dir
)
#param([Int32]$start=4000)
#param([Int32]$stop=7000)
#param([Int32]$step=100)
#param([String]$dir)

# python3 \\wsl$\Ubuntu\home\djulson\detector-readout\Neutron_CLI.py --p COM4 -e --all-sectors
# python3 \\wsl$\Ubuntu\home\djulson\detector-readout\Neutron_CLI.py --p COM4 -w -v 0 --all-sectors
# python3 \\wsl$\Ubuntu\home\djulson\detector-readout\Neutron_CLI.py --p COM4 --read --all-sectors --start 400 --stop 7000 --step 100 -d D:\B11_chip_02_pre_exposure

Write-Host "Erase all sectors"
Write-Host "python Neutron_CLI.py -p $port -e --all-sectors"

Write-Host "Erase delay 10m"
#Start-Sleep -s 600

Write-Host "Program device"
Write-Host "python Neutron_CLI.py -p $port -w -v 0 --all-sectors"

Write-Host "Program delay 1h"
#Start-Sleep -s 3600

Write-Host "Read all sectors"
python Neutron_CLI.py -p $port --read --all-sectors --start $start --stop $stop --step $step -d $dir

Write-Host "Complete"
