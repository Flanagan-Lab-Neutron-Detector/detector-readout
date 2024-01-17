import sys
import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from nisoc_readout.nr0 import ReadoutNR0
from nisoc_readout.nr1 import ReadoutNR1
from nisoc_readout.testing import ReadoutSimulator

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

def handle_vt_get_bit_count_kpage(port, base_address,read_mv):
    raise NotImplementedError("TODO(aidan): make this work")

def handle_erase_chip():
    readout.erase_chip()
    print("  Erase Chip")
    # todo: monitor erase progress

def handle_erase_sector(sector_address):
    readout.erase_sector(sector_address)
    print(f"  Erase Sector {sector_address:08X} complete")

def handle_program_sector(sector_address, prog_value):
    t = time.time()
    readout.program_sector(sector_address, prog_value)
    print(f"  Program Sector {sector_address:08X} with word {prog_value:04X}")
    # todo: monitor program progress
    print(f"  time to program sector: {time.time() - t:.2f} s")

def handle_program_chip(prog_value):
    readout.program_chip(prog_value)
    print(f"  Program Chip with word {prog_value:04X}")
    # todo: monitor program progress

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

def handle_read_data(address, read_mv, vt, chunk_size=512):
    vt_mode = True if vt is not None else False
    read_mv = read_mv if vt_mode else 0

    # Read data, unpack bits, reshape
    # Re-raise any exceptions with address and voltage information

    ret = None
    try:
        data = readout.read_data(address, chunk_size, vt_mode, read_mv)
        array = np.frombuffer(data, dtype=np.uint16)  # or dtype=np.dtype('<f4')
        bitarray = np.unpackbits(array.view(np.uint8))
        ret = np.reshape(bitarray, (chunk_size, 16))
    except Exception as e:
        raise Exception(f"Exception at {address:X}h @ {read_mv} mV.") from e

    return ret

def handle_cfg_write(address, value):
    readout.write_cfg(address, value)

def handle_cfg_read(address):
    return readout.read_cfg(address)

def handle_cfg_flash_read(offset, length, outpath, print_hex=False):
    open_flags = 'w' if print_hex else 'wb'
    outf = sys.stdout if outpath is None else open(outpath, open_flags)

    # we can read up to 4K in a single message
    nblocks = length // 4096
    nleft = length % 4096

    # enter flash passthrough
    readout.cfg_flash_enter()
    time.sleep(1)

    def write_data(data, outf, print_hex):
        if print_hex:
            # TODO: make prettier
            print(''.join([f"{b:02X}" for b in data]), file=outf)
        else:
            if outpath is None: # outf is stdout
                outf.buffer.write(data)
            else: # outf is a file opened with 'wb'
                outf.write(data)

    for bi in range(nblocks):
        data = readout.cfg_flash_read(offset + bi*4096, 4096)
        write_data(data, outf, print_hex)

    # remaining data
    data = readout.cfg_flash_read(offset + nblocks*4096, nleft)
    write_data(data, outf, print_hex)

    readout.cfg_flash_exit()

    if outpath is not None:
        outf.close()

def handle_cfg_flash_verify(binpath, flash_entry=True):
    with open(binpath, 'rb') as binfile:
        if flash_entry:
            # enter flash passthrough
            readout.cfg_flash_enter()
            time.sleep(1)
        addr = 0
        try:
            while binchunk := binfile.read(4096):
                # get data from flash
                data = readout.cfg_flash_read(addr, len(binchunk))
                # compare
                for i in range(len(binchunk)):
                    if data[i] != binchunk[i]:
                        raise Exception(f"  VERIFY FAILED at {addr+i:06X}h: flash={data[i]:02X}h file={binchunk[i]:02X}")
                # next
                addr += len(binchunk)
            print("  Flash and binfile match")
        finally:
            if flash_entry:
                readout.cfg_flash_exit()

def handle_cfg_flash_erase_chip(blocking=True):
    # enter flash passthrough
    readout.cfg_flash_enter()
    time.sleep(1)
    # erase
    print("Erase chip")
    readout.cfg_flash_erase(0, 3)
    while blocking:
        time.sleep(1)
        # monitor sr1[0] (BUSY)
        _, _, _, _, _, sr1, _, _ = readout.cfg_flash_dev_info()
        if (sr1 & 0x01) == 0:
            break
    # exit flash passthrough
    readout.cfg_flash_exit()

def cfg_flash_erase_blocking(address, erase_type):
    # assumes we've already entered flash passthrough
    # erase
    szstr = '4K' if erase_type == 0 else '32K' if erase_type == 1 else '64K' if erase_type == 2 else 'UNKNOWN SIZE'
    print(f"Erase {address:06X}h {szstr}")
    readout.cfg_flash_erase(address, erase_type)
    while True:
        time.sleep(1 if erase_type == 3 else 0.05)
        # monitor sr1[0] (BUSY)
        _, _, _, _, _, sr1, _, _ = readout.cfg_flash_dev_info()
        if (sr1 & 0x01) == 0:
            break

def handle_cfg_flash_erase_range(offset, length, entry=True):
    if entry:
        # enter flash passthrough
        readout.cfg_flash_enter()
        time.sleep(1)

    try:
        # finest erase granularity is 4K pages
        pgoff = offset // 4096
        pglen = (length + 4096-1) // 4096
        if (offset + length + 4096-1)//4096 > (offset//4096 + pglen):
            pglen += 1

        #print(f"{pgoff=} {pglen=}")

        # if we are not at a 64K boundary, erase with 32K or 4K to the next 64K block
        if (pgoff % 16) != 0:
            # if we are not at a 32K boundary, erase with 4K to the next 32K block
            while ((pgoff % 8) != 0) and (pglen > 0):
                cfg_flash_erase_blocking(pgoff*4096, 0)
                pgoff += 1
                pglen -= 1
            #print(f"{pgoff=} {pglen=}")
            # now erase a 32K block (if we have blocks remaining and not at 64K boundary)
            if (pgoff % 16 != 0) and (pglen//8 > 0):
                cfg_flash_erase_blocking(pgoff*4096, 1)
                pgoff += 8
                pglen -= 8
                #print(f"{pgoff=} {pglen=}")

        # now erase as many 64K blocks as we can
        while (pglen//16) != 0:
            cfg_flash_erase_blocking(pgoff*4096, 2)
            pgoff += 16
            pglen -= 16
            #print(f"{pgoff=} {pglen=}")

        # now erase a 32K block if we can
        if (pglen//8) != 0:
            cfg_flash_erase_blocking(pgoff*4096, 1)
            pgoff += 8
            pglen -= 8
            #print(f"{pgoff=} {pglen=}")

        # now erase 4K blocks until we're done
        while pglen > 0:
            cfg_flash_erase_blocking(pgoff*4096, 0)
            pgoff += 1
            pglen -= 1
            #print(f"{pgoff=} {pglen=}")

    finally:
        if entry:
            # exit flash passthrough
            readout.cfg_flash_exit()

def handle_cfg_flash_write(binpath):
    with open(binpath, 'rb') as binfile:
        binfilesize = os.stat(binpath).st_size
        # enter flash passthrough
        readout.cfg_flash_enter()
        # erase
        handle_cfg_flash_erase_range(0, binfilesize, entry=False)
        # now write
        addr = 0
        print()
        try:
            while binchunk := binfile.read(1024):
                # write data to flash
                print(f"\rWriting to {addr:06X} {100*(addr+len(binchunk))/binfilesize:2.0f}%", end='', flush=True)
                readout.cfg_flash_write(addr, len(binchunk), binchunk)
                while True:
                    time.sleep(0.01)
                    # monitor sr1[0] (BUSY)
                    _, _, _, _, _, sr1, _, _ = readout.cfg_flash_dev_info()
                    if (sr1 & 0x01) == 0:
                        break
                # next
                addr += len(binchunk)

            print(f"  Wrote {binpath} to flash")
            print("  Verifying...")
            handle_cfg_flash_verify(binpath, flash_entry=False)
        finally:
            readout.cfg_flash_exit()

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

    print(f"      CE 10V Counts:   {ce_10v_cts}")
    print(f"   Reset 10V Counts:   {reset_10v_cts}")
    print(f"  WP/Acc 10V Counts:   {wp_acc_10v_cts}")
    print(f"   Spare 10V Counts:   {spare_10v_cts}")

    if unit == 0:
        return 0, ce_10v_cts
    elif unit == 1:
        return 0, reset_10v_cts
    elif unit == 2:
        return 0, wp_acc_10v_cts
    elif unit == 3:
        return 0, spare_10v_cts
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
    if isinstance(readout, ReadoutNR0):
        return handle_ana_get_cal_counts_nr0(unit)
    else:
        return handle_ana_get_cal_counts_nr1(unit)

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
    if isinstance(readout, ReadoutNR0):
        handle_ana_set_cal_counts_nr0(unit, c0, c1)
    else:
        handle_ana_set_cal_counts_nr1(unit, c0, c1)

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
        NDarray = retry(3, handle_read_data, address, mv, 1, chunk_size)
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
def int_positive_auto(x, base=0) -> int:
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
read_parser.add_argument('--address', type=int_positive_auto, required=True, help='read start address')
read_parser.add_argument('--sectors', type=int_positive_auto, required=True, help='number of sectors to read')
read_parser.add_argument('--start', type=int_positive_auto, required=True, help='the lowest voltage at which to read the chip in mV')
read_parser.add_argument('--stop', type=int_positive_auto, required=True, help='the highest voltage at which to read the chip in mV')
read_parser.add_argument('--step', type=int_positive_auto, required=True, help='the granularity in mV')
read_parser.add_argument('-d', '--directory', required=True, help='folder to contain output data files (relative path)')
read_parser.add_argument('--format', type=str, choices=['binary', 'csv'], default='binary', help='select data format')
read_parser.add_argument('--chunk-size', type=int_positive_auto, default=512, help='size of read chunks; should evenly divide sector size')

erase_parser = subparsers.add_parser('erase', help='erase by sector]')
erase_parser.add_argument('--address', type=int_positive_auto, required=True, help='erase start address')
erase_parser.add_argument('--sectors', type=int_positive_auto, required=True, help='number of sectors to erase')

erase_chip_parser = subparsers.add_parser('erase-chip', help='erase entire chip')

write_parser = subparsers.add_parser('write', help='write data to the chip')
write_parser.add_argument('--address', type=int_positive_auto, required=True, help='write start address')
write_parser.add_argument('--sectors', type=int_positive_auto, required=True, help='number of sectors to write')
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
dac_activeset_parser.add_argument('value', type=int_positive_auto, help='Active value')

cfg_parser = subparsers.add_parser('cfg', help='read/write configuration registers')
cfg_subparsers = cfg_parser.add_subparsers(title="CFG commands", description="CFG commands", required=True, dest='cfg_command')
cfg_write_parser = cfg_subparsers.add_parser('write', help='write to CFG register')
cfg_write_parser.add_argument('address', type=int_positive_auto, help='cfg register address')
cfg_write_parser.add_argument('value', type=int_positive_auto, help='write value')
cfg_read_parser = cfg_subparsers.add_parser('read', help='read from CFG register')
cfg_read_parser.add_argument('address', type=int_positive_auto, help='cfg register address')

flash_parser = subparsers.add_parser('flash', help='read/write interface FPGA configuration flash')
flash_subparsers = flash_parser.add_subparsers(title='Flash commands', description='Flash commands', required=True, dest='flash_command')
flash_info_parser = flash_subparsers.add_parser('info', help='get flash info')
flash_read_parser = flash_subparsers.add_parser('read', help='read data from flash')
flash_read_parser.add_argument('-o', dest='out_path', help='output file path')
flash_read_parser.add_argument('--hex', action='store_true', help='print as ascii hex instead of binary')
flash_read_parser.add_argument('--offset', type=int_positive_auto, default=0, help='read starting at this address')
flash_read_parser.add_argument('--length', type=int_positive_auto, default=128*1024*1024, help='number of bytes to read')
flash_verify_parser = flash_subparsers.add_parser('verify', help='verify flash contents')
flash_verify_parser.add_argument('binfile', help='file to verify against')
flash_erase_chip_parser = flash_subparsers.add_parser('erase-chip', help='erase flash chip')
flash_erase_parser = flash_subparsers.add_parser('erase', help='erase flash sectors')
flash_erase_parser.add_argument('offset', type=int_positive_auto, help='erase from offset')
flash_erase_parser.add_argument('length', type=int_positive_auto, help='number of bytes to erase')
flash_write_parser = flash_subparsers.add_parser('write', help='write to flash')
flash_write_parser.add_argument('binfile', help='file to write')

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
readout = ReadoutNR0(ser_read, ser_write)

# Check if a port is valid and assign serial object if possible
# if not test, ser is assigned and readout is set to readout_prod
# if test, ser stays None and readout is set to readout_test
ser: serial.Serial = None
if(args.test):
    print("Running in Test Mode")
    simulator = ReadoutSimulator()
    def test_read(n: int) -> bytes:
        global simulator
        return simulator.ser_read(n)
    def test_write(data) -> None:
        global simulator
        simulator.ser_write(data)
    readout = ReadoutNR1(test_read, test_write)
elif(args.port):
    print("Checking status of port " + args.port)
    try:
        ser = serial.Serial(args.port, 115200, bytesize = serial.EIGHTBITS,stopbits =serial.STOPBITS_ONE, parity  = serial.PARITY_NONE,timeout=2)
    except serial.SerialException as e:
        print(e)
        parser.error("It appears there was a problem with your USB port")

    uptime, version, is_busy, *_ = readout.ping()

    if version == "NR1 test":
        print("Connected to NR1 prototype")
        readout = ReadoutNR1(ser_read, ser_write)
        uptime, version, is_busy, reset_flags, task, task_state, *_ = readout.ping()
        print("  Ping")
        print(f"    Firmware   {version}")
        print(f"    Uptime     {uptime}s")
        print(f"    Busy       {bool(is_busy)}")
        print(f"    Reset      {reset_flags:X}h")
        print(f"    Task       {task}")
        print(f"    Task State {task_state}")
    else:
        print("Connected to NR0")
        print("  Ping")
        print(f"    Firmware {version}")
        print(f"    Uptime   {uptime}s")
        print(f"    Busy     {bool(is_busy)}")

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
            _, version, *_ = readout.ping()
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
    if not os.path.exists(args.directory):
        os.mkdir(args.directory)
    print(args.directory)

    voltage_range = range(args.start, args.stop, args.step)
    sector_range = range(args.address, args.address + args.sectors*2**16, 2**16)
    write_csv = (args.format != 'binary')
    read_chip_voltages(voltage_range, sector_range, location=args.directory, csv=write_csv, chunk_size=args.chunk_size)

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

# CFG regs
elif args.command == 'cfg':
    if args.cfg_command == 'write':
        handle_cfg_write(args.address, args.value)
        print(f"  CFG reg {args.address:04X} set to {args.value:04X}")
    elif args.cfg_command == 'read':
        val = handle_cfg_read(args.address)
        print(f"  CFG reg {args.address:04X} = {val:04X}")

# Interface FPGA cfg flash
elif args.command == 'flash':
    if args.flash_command == 'info':
        # enter flash passthrough
        readout.cfg_flash_enter()
        time.sleep(1)
        # now read
        mfg, devid, jedec_type, jedec_cap, uid, sr1, sr2, sr3 = readout.cfg_flash_dev_info()
        print("Flash info")
        print(f"  mfg:        {mfg:02X}h")
        print(f"  dev id:     {devid:02X}h")
        print(f"  JEDEC type: {jedec_type:02X}h")
        print(f"  JEDEC cap:  {jedec_cap:02X}h")
        print(f"  UID:        {' '.join([f'{b:02X}h' for b in uid])}")
        print(f"  status 1:   {sr1:02X}h")
        print(f"  status 2:   {sr2:02X}h")
        print(f"  status 3:   {sr3:02X}h")
        # exit flash passthrough
        readout.cfg_flash_exit()
    elif args.flash_command == 'read':
        handle_cfg_flash_read(args.offset, args.length, args.out_path, print_hex=args.hex)
    elif args.flash_command == 'verify':
        handle_cfg_flash_verify(args.binfile)
    elif args.flash_command == 'erase-chip':
        handle_cfg_flash_erase_chip()
    elif args.flash_command == 'erase':
        handle_cfg_flash_erase_range(args.offset, args.length)
    elif args.flash_command == 'write':
        handle_cfg_flash_write(args.binfile)

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

