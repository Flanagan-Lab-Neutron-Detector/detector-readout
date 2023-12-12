import sys
import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
import numpy as np
import time
import os

import nisoc_readout as nisoc_readout

from argparse import ArgumentParser, ArgumentTypeError
from functools import partial

###########################
# Global helper functions #
###########################

# Helper function for displaying time prettily
def formatted_time(t):
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    h = t % 24
    t //= 24
    d = t % 365
    t //= 365
    y = t
    return f"{y} years {d} days" if y else f"{d} days {h} hours" if d else f"{h} hours {m} minutes" if h else f"{m} minutes {s} seconds" if m else f"{s} seconds"

# to silence warnings:
progBarStartTime = 0
# Totally unnecessary progress meter ripped from the bowels of the internet
# Print iterations progress
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = ''):
    global progBarStartTime
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    if iteration == 0:
        progBarStartTime = time.time()
        remainingTime = 0 #nobody cares
    else:
        elapsedTime = int(time.time() - progBarStartTime)
        totalTime = int(elapsedTime * total / (iteration+1))
        remainingTime = max(0, totalTime - elapsedTime)
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r\x1B[2K{prefix} |{bar}| {percent}% {suffix} / {formatted_time(remainingTime)} remaining', end = printEnd)

###########################
# Communications routines #
###########################

def handle_none(_port, _args):
    return True

def handle_vt_get_bit_count_kpage(port, base_address,read_mv):
    print("TODO(aidan): make this work")
    return False

def handle_erase_chip():
    readout.erase_chip()
    print("  Erase Chip")
    return True

def handle_erase_sector(sector_address):
    readout.erase_sector(sector_address)
    print("  Erase Sector {:08X} complete".format(sector_address))
    return True

def handle_program_sector(sector_address, prog_value):
    t = time.time()
    readout.program_sector(sector_address, prog_value)
    print("Program Sector {:08X} with word {:04X}".format(sector_address, prog_value))
    print("time to program sector: {:.2f} s".format(time.time() - t))
    return True

def handle_program_chip(prog_value):
    readout.program_chip(prog_value)
    print("  Program Chip with word {:04X}".format(prog_value))
    return True

def new_handle_read_data(address, read_mv, vt, chunk_size=512):
    vt_mode = True if vt is not None else False
    read_mv = read_mv if vt_mode else 0

    # Read data
    # Re-raise any exceptions with address and voltage information

    try:
        data = readout.read_data(address, chunk_size, vt_mode, read_mv)
    except Exception as e:
        raise Exception(f"Exception in read_data at {address:X}h @ {read_mv} mV.") from e
    return data

def handle_read_data(address, read_mv, vt, bit_mode, chunk_size=512):
    vt_mode = True if vt is not None else False
    read_mv = read_mv if vt_mode else 0

    # Read data, unpack bits, reshape
    # If bit_mode, count the number of bits set (sum the bits)
    # Re-raise any exceptions with address and voltage information

    ret = None
    try:
        data = readout.read_data(address, chunk_size, vt_mode, read_mv)
        array = np.frombuffer(data, dtype=np.uint16)  # or dtype=np.dtype('<f4')
        bitarray = np.unpackbits(array.view(np.uint8))
        if bit_mode:
            ret = sum(bitarray)
        else:
            ret = np.reshape(bitarray, (chunk_size, 16))
    except Exception as e:
        raise Exception(f"Exception at {address:X}h @ {read_mv} mV.") from e

    return ret

analog_unit_map = {
    "ce" : 0,
    "reset" : 1,
    "wp_acc" : 2,
    "spare" : 3
}

analog_unit_map_inv = {
    0 : "ce",
    1 : "reset",
    2 : "wp_acc",
    3 : "spare"
}

def handle_ana_get_cal_counts_nr0(unit: int) -> tuple[float, float]:
    ce_10v_cts, reset_10v_cts, wp_acc_10v_cts, spare_10v_cts = readout.ana_get_cal_counts()

    print("  CE 10V Counts:       {}\r\n  Reset 10V Counts:    {}\r\n  WP/Acc 10V Counts:   {}\r\n  Spare 10V Counts:    {}".format(
          ce_10v_cts, reset_10v_cts, wp_acc_10v_cts, spare_10v_cts))

    if unit == 0:
        return 0,ce_10v_cts
    elif unit == 1:
        return 0,reset_10v_cts
    elif unit == 2:
        return 0,wp_acc_10v_cts
    elif unit == 3:
        return 0,spare_10v_cts
    else:
        raise ValueError(f"Invalid unit {unit}")

def handle_ana_get_cal_counts_nr1(unit: int) -> tuple[float, float]:
    if unit == 1 or unit == 2:
        c0, c1 = readout.ana_get_cal_counts(unit)
        #print(f"  Unit {unit} {analog_unit_map_inv[2]} C0 = {c0}, C1 = {c1}")
        return c0, c1
    else:
        return None, None

def handle_ana_get_cal_counts(unit: int) -> tuple[float, float]:
    if isinstance(readout, nisoc_readout.ReadoutNR1):
        return handle_ana_get_cal_counts_nr1(unit)
    else:
        return handle_ana_get_cal_counts_nr0(unit)

def handle_ana_set_cal_counts_nr0(unit: int, c0: float, c1: float):
    # Get old values
    ce_10v_cts, reset_10v_cts, wp_acc_10v_cts, spare_10v_cts = readout.ana_get_cal_counts()
    # Update unit
    if unit == 0:
        ce_10v_cts = c1
    elif unit == 1:
        reset_10v_cts = c1
    elif unit == 2:
        wp_acc_10v_cts = c1
    elif unit == 3:
        spare_10v_cts = c1
    else:
        raise ValueError(f"Invalid unit {unit}")
    # Write back
    readout.ana_set_cal_counts(ce_10v_cts, reset_10v_cts, wp_acc_10v_cts, spare_10v_cts)
    print("  Set cal counts")

def handle_ana_set_cal_counts_nr1(unit: int, c0: float, c1: float):
    if unit == 1 or unit == 2:
        readout.ana_set_cal_counts(unit, c0, c1)
        print(f"  Set unit {unit} {analog_unit_map_inv[unit]} C0 = {c0}, C1 = {c1}")
    else:
        print(f"  Invalid unit {unit}")

def handle_ana_set_cal_counts(unit: int, c0: float, c1: float):
    if isinstance(readout, nisoc_readout.ReadoutNR1):
        handle_ana_set_cal_counts_nr1(unit, c0, c1)
    else:
        handle_ana_set_cal_counts_nr0(unit, c0, c1)

def handle_ana_set_active_counts(unit: str, counts: int):
    if unit in analog_unit_map:
        unit_num = analog_unit_map[unit]
        readout.ana_set_active_counts(unit_num, counts)
        print("  Set {} ({}) counts to {}".format(unit, unit_num, counts))
    else:
        print(f"  Unknown unit {unit}")

#####################
# Analysis routines #
#####################

# One sector is 64kword of data = 128kB
# So we need voltages * len(sectors) * 128k of storage

def retry(retries, f, *args, **kwargs):
    ret = None
    while retries > 0:
        try:
            ret = f(*args, **kwargs)
            break
        except Exception as e:
            print(f"\n\nException. Retrying: {e}")
            if retries == 1:
                print("\nRetries exceeded.")
                raise e
        finally:
            retries -= 1
    return ret

def read_sector_bin(mv, sa, location, chunk_size=512):
    # read 512-word (1024-byte) chunks
    addr_range = range(sa, sa + 65536, chunk_size)
    mem: list[bytearray] = [bytearray()]*len(addr_range)
    for i,address in enumerate(addr_range):
        #mem[i] = read_chunk_retries_bin(mv, address, retries=3)
        mem[i] = retry(3, new_handle_read_data, address, mv, True, chunk_size)
    # write sector-voltage data to disk
    with open(os.path.join(location, f"data-{mv}-{sa}.bin"), 'ab') as f:
        for block in mem:
            f.write(block)

def read_sector_csv(mv, sa, location, chunk_size=512):
    saved_array = np.zeros((chunk_size,16))
    for idx, address in enumerate(range(sa, sa + 65536, chunk_size)):
        #NDarray = read_chunk_retries_csv(mv, address, retries=3)
        NDarray = retry(3, handle_read_data, address, mv, 1, 0, chunk_size)
        if idx == 0:
            saved_array = NDarray
        else:
            saved_array = np.vstack((saved_array,NDarray))
    np.savetxt(os.path.join(location, f"data-{mv}-{sa}.csv"), saved_array, fmt='%d', delimiter='')

# read voltage from the chip
def read_chip_voltages(voltages, sectors, location='.', csv=False, chunk_size=512):
    # Progress bar
    count = 1
    printProgressBar(0, len(voltages)*len(sectors), prefix='Getting sweep data: ', suffix='complete', decimals=0, length=50)

    # Loop through sector base addresses
    for _, base_address in enumerate(sectors):
        # Loop through read voltages
        for _, mv in enumerate(voltages):
            if csv:
                read_sector_csv(mv, base_address, location, chunk_size=chunk_size)
            else:
                read_sector_bin(mv, base_address, location, chunk_size=chunk_size)
            count+=1
            printProgressBar(count, len(voltages)*len(sectors), prefix='Getting sweep data: ', suffix='complete', decimals=0, length=50)
    print()

#######
# CLI #
#######

# range-restricted int conversion
def int_range(x, min: int, max: int, base=10) -> int:
    x = int(x, base=base)
    if x < min or x > max:
        raise ArgumentTypeError(f"{x} must be in range [{min}, {max}]")
    return x

# positive int conversion
def int_positive(x, base=10) -> int:
    x = int(x, base=base)
    if x < 0:
        raise ArgumentTypeError(f"{x} must be positive")
    return x

# CLI spec
parser = ArgumentParser(description="CLI for chip reads/writes")

parser.add_argument('-p', '--port', help='the USB port where the chip reader is installed')

parser.add_argument('-t', '--test', action='store_true', help='bypass real serial port')

subparsers = parser.add_subparsers(title="command", description="commands", dest='command')

port_parser = subparsers.add_parser('list', help='list serial ports')

read_parser = subparsers.add_parser('read', help='read data from the chip')
read_parser.add_argument('--address', type=partial(int_positive, base=0), required=True, help='read start address')
read_parser.add_argument('--sectors', type=partial(int_positive, base=0), required=True, help='number of sectors to read')
read_parser.add_argument('--start', type=partial(int_positive, base=0), required=True, help='the lowest voltage at which to read the chip in mV')
read_parser.add_argument('--stop', type=partial(int_positive, base=0), required=True, help='the highest voltage at which to read the chip in mV')
read_parser.add_argument('--step', type=partial(int_positive, base=0), required=True, help='the granularity in mV')
read_parser.add_argument('-d', '--directory', required=True, help='folder to contain output data files (relative path)')
read_parser.add_argument('--format', type=str, choices=['binary', 'csv'], default='binary', help='select data format')
read_parser.add_argument('--chunk-size', type=partial(int_positive, base=0), default=512, help='size of read chunks; should evenly divide sector size')

erase_parser = subparsers.add_parser('erase', help='erase by sector]')
erase_parser.add_argument('--address', type=partial(int_positive, base=0), required=True, help='erase start address')
erase_parser.add_argument('--sectors', type=partial(int_positive, base=0), required=True, help='number of sectors to erase')

erase_chip_parser = subparsers.add_parser('erase-chip', help='erase entire chip')

write_parser = subparsers.add_parser('write', help='write data to the chip')
write_parser.add_argument('--address', type=partial(int_positive, base=0), required=True, help='write start address')
write_parser.add_argument('--sectors', type=partial(int_positive, base=0), required=True, help='number of sectors to write')
write_parser.add_argument('-v', '--value', type=partial(int_range, min=0, max=65535, base=0), required=True, help='the value to store in each word')

write_chip_parser = subparsers.add_parser('write-chip', help='write data to the entire chip')
write_chip_parser.add_argument('-v', '--value', type=partial(int_range, min=0, max=65535, base=0), required=True, help='the value to store in each word')

dac_parser = subparsers.add_parser('dac', help='DAC management')
dac_subparsers = dac_parser.add_subparsers(title="DAC commands", description="DAC commands", required=True, dest='dac_command')
dac_calget_parser = dac_subparsers.add_parser('get-calibration', help='get DAC calibration values')
dac_calset_parser = dac_subparsers.add_parser('set-calibration', help='set DAC calibration values')
dac_calset_parser.add_argument('unit', type=int, choices=[0, 1, 2, 3], help='DAC unit')
dac_calset_parser.add_argument('--c0', type=float, required=True, help='Calibration C0 (offset) value')
dac_calset_parser.add_argument('--c1', type=float, required=True, help='Calibration C1 (gain) value')
dac_activeset_parser = dac_subparsers.add_parser('set-active', help='set DAC active values')
dac_activeset_parser.add_argument('unit', type=int, choices=[0, 1, 2, 3], help='DAC unit')
dac_activeset_parser.add_argument('value', type=partial(int_positive, base=0), help='Active value')

args = parser.parse_args()

# Serial read/write
class SerialTimeoutError(Exception):
    """Raised on timeout when reading serial port"""
    pass

def ser_read(n: int) -> bytes:
    data = ser.read(n)
    if len(data) < n:
        raise SerialTimeoutError(f"Serial timeout. Expected {n} bytes, got {len(data)}")
    return data

def ser_write(data) -> None:
    time.sleep(0.003)
    ser.write(data)

# initialize readout
readout = nisoc_readout.ReadoutNR0(ser_read, ser_write)

# Check if a port is valid and assign serial object if possible
# if not test, ser is assigned and readout is set to readout_prod
# if test, ser stays None and readout is set to readout_test
ser: serial.Serial = None
if(args.test):
    print("Running in Test Mode")
    readout = nisoc_readout.ReadoutDummy(ser_read, ser_write)
elif(args.port):
    print("Checking status of port " + args.port)
    try:
        ser = serial.Serial(args.port, 115200, bytesize = serial.EIGHTBITS,stopbits =serial.STOPBITS_ONE, parity  = serial.PARITY_NONE,timeout=1)
    except serial.SerialException as e:
        print(e)
        parser.error("It appears there was a problem with your USB port")

    uptime, version, is_busy = readout.ping()

    if version == "NR1 test":
        print("Connected to NR1 prototype")
        readout = nisoc_readout.ReadoutNR1(ser_read, ser_write)
        uptime, version, is_busy = readout.ping()
    else:
        print("Connected to NR0")

    print("  Ping")
    print(f"    Firmware {version}")
    print(f"    Uptime   {uptime}s")
    print(f"    idle     {bool(is_busy)}")
else:
    print("Neither -t nor -p specified. Good luck.")

# List ports
if args.command == 'list':
    ports = serial.tools.list_ports.comports()

    nisoc_ports = []
    other_ports = []
    for port in ports:
        try:
            ser = serial.Serial(port[0], 115200, bytesize=serial.EIGHTBITS, stopbits=serial.STOPBITS_ONE, parity=serial.PARITY_NONE, timeout=0.05)
            _, version, _ = readout.ping()
            # if no exceptions, it's a working NRx
            nisoc_ports.append((port[0], version))
        except:
            other_ports.append(port)

    if len(nisoc_ports) > 0:
        print()
        print("NISoC Readouts:")
        for port, version in sorted(nisoc_ports):
            print(f"\t{port} {version}")
    if len(other_ports) > 0:
        print()
        print("Other serial ports:")
        for port, desc, _ in sorted(other_ports):
            print(f"\t{port} {desc}")
    if len(nisoc_ports) == 0 and len(other_ports) == 0:
        print("No serial ports found")

# Read a sector or entire chip
elif args.command == 'read':
    if(args.directory and not os.path.exists(args.directory)):
        os.mkdir(args.directory)
    directory = args.directory if args.directory else '.'
    print(directory)

    voltage_range = range(args.start, args.stop, args.step)
    sector_range = range(args.address, args.address + args.sectors*2**16, 2**16)
    write_csv = (args.format != 'binary')
    read_chip_voltages(voltage_range, sector_range, location=directory, csv=write_csv, chunk_size=args.chunk_size)

# Erase whole chip
elif args.command == 'erase-chip':
    handle_erase_chip()

# Erase sectors
elif args.command == 'erase':
    sector_range = range(args.address, args.address + args.sectors*2**16, 2**16)
    for sector in sector_range:
        handle_erase_sector(sector)
        time.sleep(0.5)

# Write entire chip
elif args.command == 'write-chip':
    try:
        handle_program_chip(args.value)
    except SerialTimeoutError as te:
        print("Writing a chip takes about 30 minutes and will continue despite the serial error that causes this macro to abort. Please wait until the chip stops flashing.")
        print(te)
    except Exception as e:
        print(f"Got unexpected exception when programming chip: {e}")

# Write sectors
elif args.command == 'write':
    sector_range = range(args.address, args.address + args.sectors*2**16, 2**16)
    for sector in sector_range:
        handle_program_sector(sector, args.value)

# Print calibration counts
elif args.command == 'dac' and args.dac_command == 'get-calibration':
    for unit in range(4):
        c0, c1 = handle_ana_get_cal_counts(unit)
        print(f"  DAC unit {unit} ({analog_unit_map_inv[unit]}): C0 = {c0}, C1 = {c1}")

elif args.command == 'dac' and args.dac_command == 'set-calibration':
    # Get old calibration
    old_c0, old_c1 = handle_ana_get_cal_counts(args.unit)

    # Set new calibration
    handle_ana_set_cal_counts(args.unit, args.c0, args.c1)

# Set analog output
elif args.command == 'dac' and args.dac_command == 'set-active':
    # counts and unit are verified above
    readout.ana_set_active_counts(args.unit, args.value)
    print(f"  Set unit {args.unit} ({analog_unit_map_inv[args.unit]}) counts to {args.value}")

