# detector-readout

Utilities for interfacing with NISoC readout boards.

## Setup

### Linux/Mac

Prerequisites: Python 3.9 or later.

```bash
git clone https://github.com/Flanagan-Lab-Neutron-Detector/detector-readout.git
cd detector-readout
python3 -m venv .venv
source ./.venv/bin/activate
python3 -m pip install -r requirements.txt
```

### Windows

Prerequisites: Python 3.9 or later, Visual Studio tools.

```pwsh
git clone "https://github.com/Flanagan-Lab-Neutron-Detector/detector-readout.git"
cd detector-readout
python3.exe -m venv .venv
.venv\Scripts\Activate.ps1
python3.exe -m pip install -r requirements.txt
```

## Usage

`Neutron_CLI.py` is the preferred utility to communicate with NISoC readouts. The standard preread procedure can be run with `epr_chip.sh` (`EraseProgramRead.ps1` for PowerShell)

Run `python3 Neutron_CLI.py -h` for a list of commands. The most common operations are:

### List Serial Ports
```
python3 Neutron_CLI.py list
```

### Read Whole Chip from 1 to 8 V in 100 mV Steps
```
python3 Neutron_CLI.py -p <port> read --address 0 --sectors 1024 --start 1000 --stop 8000 --step 100 -d <output directory>
```

## Notes

`Neutron_CLI.py` is the preferred utility to communicate with the readout. `Neutron_GUI.py` is a graphical alternative. `nisoc_readout.py` is a Python module implementing the communications protocol and available commands. `nisoc_readout_dummy.py` contains stub functions for testing readout front-ends.
