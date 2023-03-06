Write-Host "Erase all sectors"
python3 \\wsl$\Ubuntu\home\djulson\detector-readout\Neutron_CLI.py --p COM4 -e --all-sectors
Write-Host "Erase delay 10m"
Start-Sleep -s 600
Write-Host "Program device"
python3 \\wsl$\Ubuntu\home\djulson\detector-readout\Neutron_CLI.py --p COM4 -w -v 0 --all-sectors
Write-Host "Program delay 1h"
Start-Sleep -s 3600
Write-Host "Read all sectors"
python3 \\wsl$\Ubuntu\home\djulson\detector-readout\Neutron_CLI.py --p COM4 --read --all-sectors --start 400 --stop 7000 --step 100 -d D:\B11_chip_02_pre_exposure
Write-Host "Complete"