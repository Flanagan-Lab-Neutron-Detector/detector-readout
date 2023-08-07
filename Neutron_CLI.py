import sys
import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
import numpy as np
import time
import os

import nisoc_readout as readout_prod
import nisoc_readout_dummy as readout_test

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

    # cmd_len = 16
    # rsp_len = 1032
    # cmd_data = make_cmd(cmd_len, 6)
    # struct.pack_into("<II", cmd_data, 4, base_address, read_mv)
    # insert_crc(cmd_data)

    # port.write(cmd_data)
    # rsp_data = read_rsp(port, rsp_len, 7)


    # if len(rsp_data) >= rsp_len:
    #     array = np.frombuffer(rsp_data[4:-4], dtype=np.uint8)  # or dtype=np.dtype('<f4')
    #     bitarray = np.unpackbits(array)#np.unpackbits(array.view(np.uint8))

    #     NDarray = np.reshape(bitarray, (1024, 8))   #np.reshape(bitarray, (512, 16)) #if we are reading 16 bit words

    # else:
    #     return False
    #return NDarray
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

def new_handle_read_data(address, read_mv, vt):
    vt_mode = True if vt is not None else False
    read_mv = read_mv if vt_mode else 0

    data = readout.read_data(address, vt_mode, read_mv)
    return data

def handle_read_data(address, read_mv, vt, bit_mode):
    vt_mode = True if vt is not None else False
    read_mv = read_mv if vt_mode else 0

    # Original adapted for read_data call.
    # I don't know what this is doing or why.
    # Removed repeat loop. Callers should catch exceptions

    data = readout.read_data(address, vt_mode, read_mv)

    array = np.frombuffer(data, dtype=np.uint16)  # or dtype=np.dtype('<f4')
    bitarray = np.unpackbits(array.view(np.uint8))
    if bit_mode:
        return sum(bitarray)
    else:
        NDarray = np.reshape(bitarray, (512, 16))
        return NDarray

    # Original for posterity

    # if len(rsp_data) >= rsp_len:
    #     array = np.frombuffer(rsp_data[4:-4], dtype=np.uint16)  # or dtype=np.dtype('<f4')
    #     bitarray = np.unpackbits(array.view(np.uint8))
    #     NDarray = np.reshape(bitarray, (512, 16))
    #     if(current_iter > 1):
    #         print(f"Read successful on attempt {current_iter}")
    #      #if we are reading 16 bit words
    #     # for line in range(32):
    #     #      print("  {:08X}    {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X}".format(base_address + 16*line,
    #     #                                                                                                                                                 rsp_data[4+16*line + 0], rsp_data[4+16*line + 1], rsp_data[4+16*line + 2], rsp_data[4+16*line + 3],
    #     #                                                                                                                                                 rsp_data[4+16*line + 4], rsp_data[4+16*line + 5], rsp_data[4+16*line + 6], rsp_data[4+16*line + 7],
    #     #                                                                                                                                                 rsp_data[4+16*line + 8], rsp_data[4+16*line + 8], rsp_data[4+16*line + 8], rsp_data[4+16*line + 9],
    #     #                                                                                                                                                 rsp_data[4+16*line + 12], rsp_data[4+16*line + 13], rsp_data[4+16*line + 14], rsp_data[4+16*line + 15]))
    #     # #print(read_mv,sum(bitarray))
    # else:
    #     # repeat call
    #     if current_iter > max_repeats:
    #         print(f"Unable to read after {max_repeats} attempts. Skipping.")
    #         return 0 if bit_mode else np.full((512, 16),2,dtype=np.uint8)
    #     else:
    #         print(f"Repeating call to read data at address {address}. Attempt #{current_iter} failed.")
    #         return handle_read_data(port, address,read_mv,vt,bit_mode, max_repeats=max_repeats, current_iter=current_iter+1)
    # if bit_mode:
    #     return sum(bitarray)
    # else:
    #     return NDarray

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
    if isinstance(readout, readout_prod.ReadoutNR1):
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
    if isinstance(readout, readout_prod.ReadoutNR1):
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


def hex_int(x):
    return int(x, 16)


def rev_b_get_dac_code(mv):
    # Dac is 12-bit 0V-3.3V with an amplifier with G = 7.667
    # dac output = code * 3.3/4095
    # amp output = dac * 7.667
    # mv = G * (code/4095) * 3.3V   =>   code = 4095 * mv/(G*3.3V)
    return 4095 * int(mv/(7.667*3.3))
    #gain = 10/1627*4095/3.3
    #return int(mv/gain*4095/3.3)

def rev_b_get_mv(dac_code):
    # Dac is 12-bit 0V-3.3V with an amplifier with G = 7.667
    # dac output = code * 3.3/4095
    # amp output = dac * 7.667
    return 7.667 * float(dac_code/4095) * 3.3
    #gain = 10/1627*4095/3.3
    #return float(gain*3.3/4095*dac_code)

# One sectors is 64kword of data = 128kB
# So we need voltages * len(sectors) * 128k of storage

def read_chip_voltages_binary(voltages, sectors, location='.'):
    count = 1
    printProgressBar(0, len(voltages)*128*len(sectors), prefix='Getting sweep data: ', suffix='complete', decimals=0, length=50)
    for _, base_address in enumerate(sectors):
        for _, j in enumerate(voltages):
            #mem = [bytearray() for _ in range(128)]
            #for idx, address in enumerate(range(base_address, base_address + 65024 + 512, 512)):
            #    mem[idx] = handle_read_data(address, j, 1, 0)
            addr_range = range(base_address, base_address + 65024 + 512, 512)
            mem: list[bytearray] = [bytearray()]*len(addr_range)
            for i,address in enumerate(addr_range):
                #print(f"\n[read_chip_voltages_binary] Reading {base_address:X} + {address-base_address:X} @ {j} mV")
                data = []
                retries = 3
                while retries > 0:
                    try:
                        data = new_handle_read_data(address, j, True)
                        break
                    except Exception as e:
                        print(f"\n\nException in read_chip_voltages at {address} @ {j} mV: {e}\n")
                    finally:
                        retries -= 1
                if retries == 0:
                    print("\nRetries exceeded. Exiting.")
                    exit(1)

                #mem.append(data)
                mem[i] = data
                count+=1
            printProgressBar(count, len(voltages)*128*len(sectors), prefix='Getting sweep data: ', suffix='complete', decimals=0, length=50)
            #mem = [new_handle_read_data(address, j, 1, 0)
            #       for _,address in enumerate(range(base_address, base_address + 65024 + 512, 512))]
            with open(os.path.join(location, f"data-{j}-{base_address}.bin"), 'ab') as f:
                for block in mem:
                    f.write(block)
    print()

def read_chip_voltages_csv(voltages, sectors, location='.'):
    count = 1
    printProgressBar(0, len(voltages)*128*len(sectors), prefix='Getting sweep data: ', suffix='complete', decimals=0, length=50)
    for idx2, base_address in enumerate(sectors):
        for idx1, j in enumerate(voltages):
            saved_array = np.zeros((512,16))
            for idx, address in enumerate(range(base_address, base_address + 65024 + 512, 512)):
                NDarray = None

                retries = 3
                while retries > 0:
                    try:
                        NDarray = handle_read_data(address, j, 1, 0)
                        break
                    except Exception as e:
                        print(f"Exception in read_chip_voltages at {address} @ {j} mV: {e}")
                    finally:
                        retries -= 1
                if retries == 0:
                    print("Retries exceeded. Exiting")
                    exit(1)

                if idx == 0:
                    saved_array = NDarray
                else:
                    saved_array = np.vstack((saved_array,NDarray))
                    #print(saved_array)
                printProgressBar(count, len(voltages)*128*len(sectors), prefix='Getting sweep data: ', suffix='complete', decimals=0, length=50)
                count+=1
            np.savetxt(os.path.join(location,"data-{}-{}.csv".format(j,base_address)), saved_array, fmt='%d', delimiter='')

    print()
    return

def count_chip_bits(voltages, sectors, location='.'):
    progress_count = 1
    printProgressBar(0, len(voltages)*len(sectors), prefix='Getting sweep data: ', suffix='complete', decimals=0, length=50)
    for _, voltage in enumerate(voltages):
        counts: list[int] = []
        #print(f"{voltage}mV")
        #print("SectorAddress, Count")
        for _, base_address in enumerate(sectors):
            count = readout.get_sector_bit_count(base_address, voltage)
            counts.append(count)
            #print(f"0x{base_address:X}, {count:d}")
            printProgressBar(progress_count, len(voltages)*len(sectors), prefix='Getting sweep data: ', suffix='complete', decimals=0, length=50)
            progress_count+=1
        with open(os.path.join(location,"counts-{}.csv".format(voltage)), 'w') as f:
            f.write("SectorAddress, Count\n")
            for i in range(len(sectors)):
                f.write(f"0x{sectors[i]:X}, {counts[i]:d}\n")
    print()

# ## No longer used
# def get_voltage_sweep(port,base_address,voltages,method):
#     if not method:
#         values = []
#         tracker = 0
#         printProgressBar(0, len(voltages)*128, prefix='Getting sweep data: ', suffix='complete', decimals=0, length=50)
#         for idx1,j in enumerate(voltages):
#             temp = []
#             for idx, address in enumerate(range(base_address, base_address + 65024 + 512, 512)):
#                 NDarray = handle_read_data(port, address, j,1,1)
#                 if not isinstance(NDarray,bool):
#                     temp.append(NDarray)
#                 printProgressBar(tracker+1, len(voltages)*128, prefix='Getting sweep data: ', suffix='complete', decimals=0, length=50)
#                 tracker+=1
#             values.append([j, sum(temp)])

#         v = [x[0] for x in values]
#         bits = [x[1] for x in values]
#     else:
#         bits = []
#         v = voltages
#         printProgressBar(0, len(voltages), prefix='Getting sweep data: ', suffix='complete', decimals=0, length=50)
#         for idx, j in enumerate(voltages):
#             bit_count = handle_get_sector_bit_count(port,base_address,read_mv=j)
#             bits.append(bit_count)
#             time.sleep(.003)
#             printProgressBar(idx, len(voltages), prefix='Getting sweep data: ', suffix='complete', decimals=0, length=50)
#     if plt.fignum_exists(1):
#         ax1 = plt.figure(1).axes[0]
#         ax2 = plt.figure(1).axes[1]
#     else:
#         fig1 = plt.figure(1)
#         ax1 = fig1.gca()
#         ax1.set_xlabel('Voltage (mV)')
#         ax1.set_ylabel('Counts')
#         ax2 = ax1.twinx()
#         ax2.set_ylabel('Sector Dist.')
#     #print("v: ")
#     #print(v)
#     #print("bits: ")
#     #print(bits)
#     ax1.plot(v, bits, label='Sector: {}'.format(base_address))
#     ax2.plot(v[1:], np.diff(bits), '--', label='Sector Dist.: {}, {}'.format(base_address, voltages[2] - voltages[1]))
#     ax2.legend(loc='best')
#     ax1.legend(loc='best')
#     plt.show()
#     return

def get_bit_noise(sector_address, voltages,iterations):
    max = iterations
    real_count = 0
    n_bins = int((voltages[-1]-voltages[0])/10)
    step = voltages[2]-voltages[1]
    results_matrix = []
    bar = 0
    for i in range(max):
        data_matrix = []
        for idx,j in enumerate(voltages):
            #print(idx,i,j)
            NDarray = handle_read_data(sector_address, j,1,0)
            if not isinstance(NDarray,bool):
                data_matrix.append([j,NDarray])
                # vgrad = np.gradient(NDarray)
                # xgrad = vgrad[0]
                # x, y = range(0, xgrad.shape[0]), range(0, xgrad.shape[1])
                # #xi, yi = np.meshgrid(x, y)
                #rbf = scipy.interpolate.Rbf(xi, yi, xgrad)
                # hf = plt.figure()
                # ha = hf.add_subplot(111, projection='3d')
                # X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
                # print(xgrad.shape,X.shape,Y.shape)
                # ha.plot_surface(X.T, Y.T, xgrad)
                # plt.show()
            # TODO(aidan): below line does not work. Replce "sg" progress meter with actual progress meter
            #sg.one_line_progress_meter('Getting Noise Data', bar + 1, len(voltages)*iterations)
            bar+=1
        results_matrix.append(data_matrix)
    true_values = []
    ix = 304
    iy = 7
    for sweep in results_matrix:
        previous_value = 0
        true_value = 0
        # for ix, iy in np.ndindex(sweep[0][1].shape):
        for idx,[mv,matrix] in enumerate(sweep):
            value = matrix[ix,iy]*mv
            if previous_value == 0 and value >0:
                true_value = value - step/2
                true_values.append(true_value)
            previous_value = value
    plt.hist(true_values,bins=n_bins)
    print(np.mean(true_values),np.std(true_values))
    plt.show()

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

# Check if a port is valid and assign serial object if possible
# if not test, ser is assigned and readout is set to readout_prod
# if test, ser stays None and readout is set to readout_test
ser: serial.Serial = None
readout = None
if(args.test):
    print("Running in Test Mode")
    readout = readout_test.Readout(ser_read, ser_write)
elif(args.port):
    readout = readout_prod.ReadoutNR0(ser_read, ser_write)
    print("Checking status of port " + args.port)
    try:
        ser = serial.Serial(args.port, 115200, bytesize = serial.EIGHTBITS,stopbits =serial.STOPBITS_ONE, parity  = serial.PARITY_NONE,timeout=1)
    except serial.SerialException as e:
        print(e)
        parser.error("It appears there was a problem with your USB port")

    uptime, version, is_busy = readout.ping()

    if version == "NR1 test":
        print("Connected to NR1 prototype")
        readout = readout_prod.ReadoutNR1(ser_read, ser_write)
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
    port_names = []
    for port, desc, hwid in sorted(ports):
        print(port, desc)

# Read a sector or entire chip
elif args.command == 'read':
    if(args.directory and not os.path.exists(args.directory)):
        os.mkdir(args.directory)
    directory = args.directory if args.directory else '.'
    print(directory)

    voltage_range = range(args.start, args.stop, args.step)
    sector_range = range(args.address, args.address + args.sectors*2**16, 2**16)
    if args.format == 'binary':
        read_chip_voltages_binary(voltage_range, sector_range, location=directory)
    else:
        read_chip_voltages_csv(voltage_range, sector_range, location=directory)

# Erase whole chip
elif args.command == 'erase-chip':
    handle_erase_chip()

# Erase sectors
elif args.command == 'erase':
    sector_range = range(args.address, args.address + args.sectors*2**16, 2**16)
    for sector in sector_range:
        handle_erase_sector(sector)

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

