import sys
import crcmod
import serial
import serial.tools.list_ports
import struct
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy
#import PySimpleGUI as sg      
import os

from argparse import ArgumentParser
from functools import partial
from datetime import timedelta


crc_poly = 0x104C11DB7  # 0x11EDC6F41
crc_seed = 0x100000007 # 0xB319AFF0
#crc_seed = 0xFFFFFFFF
xor_out = 0
#xor_out = 0xFFFFFFFF
# CRCs are compatible with Renesas Synergy CRC hardware used the dumb way
# for CRC32-C: seed=0, xor_out=0, poly=0x11EDC6F41, reverse=false, reverse order of each 4-byte chunk
# for CRC32: seed=(seed), xor_out=0, poly=0x104C11DB7, reverse=false, reverse order of each 4-byte chunk

#from tqdm import tqdm
crc_f = crcmod.mkCrcFun(crc_poly, rev=False, initCrc=crc_seed, xorOut=xor_out)

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
    return f"{y} years" if y else f"{d} days" if d else f"{h} hours" if h else f"{m} minutes" if m else f"{s} seconds"

# Totally unnecessary progress meter ripped from the bowels of the internet
# Print iterations progress
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
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
        remainingTime = totalTime - elapsedTime
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix} / {formatted_time(remainingTime)} remaining        ', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
        
def crc_4(arr):
    crc = crc_seed
    i = 0
    while i < len(arr):
        crc = crc_f(bytearray((arr[i+3], arr[i+2], arr[i+1], arr[i])), crc)
        i += 4
    return crc

hdr_start_char = 0x7E

def msg_is_valid(data, exp_id):
    start_char, length, id = struct.unpack_from("<BHB", data, 0)
    #print(start_char, length, id)
    if start_char != 0x7E:
        print("wrong start char")
        return False
    if len(data) < length:
        print("data length too short")
        return False
    if id != exp_id:
        print("wrong id")
        return False
    msg_crc, = struct.unpack_from("<I", data, length-4)
    crc = crc_4(data[0:length-4])
    if msg_crc != crc:
        print("msg crc {} != calc crc {}".format(msg_crc, crc))
        return False
    return True

def make_cmd(cmd_len, id):
    data = bytearray(cmd_len)
    struct.pack_into("<BHB", data, 0, hdr_start_char, cmd_len, id)
    return data

def insert_crc(data):
    crc = crc_4(data[0:len(data)-4])
    struct.pack_into("<I", data, len(data)-4, crc)

def read_rsp(port, rsp_len, rsp_id):
    rsp_data = port.read(rsp_len)
    if len(rsp_data) >= rsp_len:
        if msg_is_valid(rsp_data, rsp_id):
            #print(rsp_data)
            return rsp_data
        else:
            print("! Invalid response {}".format(rsp_data))
            return bytearray()
    else:
        print("! Error: Serial timeout")
        #print(rsp_data)
        #print(bytearray())
        return bytearray()

def handle_none(_port, _args):
    return True

def handle_ping(port, args):
    cmd_len = 8
    rsp_len = 32
    cmd_data = make_cmd(cmd_len, 2)

    insert_crc(cmd_data)

    #print(' '.join("{:02X}".format(B) for B in cmd_data))

    port.write(cmd_data)
    rsp_data = read_rsp(port, rsp_len, 3)
    if len(rsp_data) >= rsp_len:
        uptime, version_bin, is_busy = struct.unpack_from("<I16sI", rsp_data, 4)
        version = version_bin.decode()
    else:
        return False
    return [uptime, version, is_busy]

def handle_vt_get_bit_count_kpage(port, base_address,read_mv):

    cmd_len = 16
    rsp_len = 1032
    cmd_data = make_cmd(cmd_len, 6)
    struct.pack_into("<II", cmd_data, 4, base_address, read_mv)
    insert_crc(cmd_data)

    port.write(cmd_data)
    rsp_data = read_rsp(port, rsp_len, 7)


    if len(rsp_data) >= rsp_len:
        array = np.frombuffer(rsp_data[4:-4], dtype=np.uint8)  # or dtype=np.dtype('<f4')
        bitarray = np.unpackbits(array)#np.unpackbits(array.view(np.uint8))

        NDarray = np.reshape(bitarray, (1024, 8))   #np.reshape(bitarray, (512, 16)) #if we are reading 16 bit words

        # for line in range(63):
        #     print(line,4+64*line,4+64*line+15)
        #     print("  {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X}".format(rsp_data[4+64*line + 0], rsp_data[4+64*line + 1], rsp_data[4+64*line + 2], rsp_data[4+64*line + 3],
        #                                                                                                                                      rsp_data[4+64*line + 4], rsp_data[4+64*line + 5], rsp_data[4+64*line + 6], rsp_data[4+64*line + 7],
        #                                                                                                                                      rsp_data[4+64*line + 8], rsp_data[4+64*line + 9], rsp_data[4+64*line +10], rsp_data[4+64*line +11],
        #                                                                                                                                      rsp_data[4+64*line +12], rsp_data[4+64*line +13], rsp_data[4+64*line +14], rsp_data[4+64*line +15]))

    else:
        return False
    return NDarray

def handle_erase_chip(port):
    cmd_len = 8
    rsp_len = 8
    cmd_data = make_cmd(cmd_len, 8)
    insert_crc(cmd_data)

    port.write(cmd_data)
    rsp_data = read_rsp(port, rsp_len, 9)
    if len(rsp_data) >= rsp_len:
        print("  Erase Chip")
    else:
        return False
    return True

def handle_erase_sector(port,sector_address):
    cmd_len = 12
    rsp_len = 8
    cmd_data = make_cmd(cmd_len, 10)
    struct.pack_into("<I", cmd_data, 4, sector_address)
    insert_crc(cmd_data)

    port.write(cmd_data)
    rsp_data = read_rsp(port, rsp_len, 11)
    if len(rsp_data) >= rsp_len:
        print("  Erase Sector {:08X} complete".format(sector_address))
    else:
        return False
    return True

def handle_program_sector(port, sector_address,prog_value):
    t = time.time()
    cmd_len = 16
    rsp_len = 8
    cmd_data = make_cmd(cmd_len, 12)
    struct.pack_into("<IHH", cmd_data, 4, sector_address, prog_value, 0)
    insert_crc(cmd_data)

    port.write(cmd_data)
    rsp_data = read_rsp(port, rsp_len, 13)
    if len(rsp_data) >= rsp_len:
        print("Program Sector {:08X} with word {:04X}".format(sector_address, prog_value))
    else:
        return False
    print("time to program sector: {:.2f} s".format(time.time() - t))
    return True

def handle_program_chip(port, prog_value):
    cmd_len = 12
    rsp_len = 8
    cmd_data = make_cmd(cmd_len, 14)
    struct.pack_into("<HH", cmd_data, 4, prog_value, 0)
    insert_crc(cmd_data)

    port.write(cmd_data)
    rsp_data = read_rsp(port, rsp_len, 15)
    if len(rsp_data) >= rsp_len:
        print("  Program Chip with word {:04X}".format(prog_value))
    else:
        return False
    return True

def handle_get_sector_bit_count(port, sector_address,read_mv):
    cmd_len = 16
    rsp_len = 12
    cmd_data = make_cmd(cmd_len, 16)
    struct.pack_into("<II", cmd_data, 4, sector_address, read_mv)
    insert_crc(cmd_data)

    port.write(cmd_data)
    rsp_data = read_rsp(port, rsp_len, 17)
    if len(rsp_data) >= rsp_len:
        bits_set, = struct.unpack_from("<I", rsp_data, 4)
        #print"Sector: {}\nVoltage: {}mV\nCounts: {}\n".format(sector_address, read_mv, bits_set))
    else:
        return False
    #print(bits_set)
    return bits_set

def handle_read_data(port, address,read_mv,vt,bit_mode, max_repeats=3, current_iter=1):
    vt_mode = 1 if vt is not None else 0
    read_mv = read_mv if vt_mode else 0

    cmd_len = 20
    rsp_len = 1032

    cmd_data = make_cmd(cmd_len, 18)
    struct.pack_into("<III", cmd_data, 4, address, vt_mode, read_mv)
    insert_crc(cmd_data)

    port.write(cmd_data)
    rsp_id = 19
    rsp_data = read_rsp(port, rsp_len, rsp_id)


    if len(rsp_data) >= rsp_len:
        array = np.frombuffer(rsp_data[4:-4], dtype=np.uint16)  # or dtype=np.dtype('<f4')
        bitarray = np.unpackbits(array.view(np.uint8))
        NDarray = np.reshape(bitarray, (512, 16))
        if(current_iter > 1):
            print(f"Read successful on attempt {current_iter}")
         #if we are reading 16 bit words
        # for line in range(32):
        #      print("  {:08X}    {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X}".format(base_address + 16*line,
        #                                                                                                                                                 rsp_data[4+16*line + 0], rsp_data[4+16*line + 1], rsp_data[4+16*line + 2], rsp_data[4+16*line + 3],
        #                                                                                                                                                 rsp_data[4+16*line + 4], rsp_data[4+16*line + 5], rsp_data[4+16*line + 6], rsp_data[4+16*line + 7],
        #                                                                                                                                                 rsp_data[4+16*line + 8], rsp_data[4+16*line + 8], rsp_data[4+16*line + 8], rsp_data[4+16*line + 9],
        #                                                                                                                                                 rsp_data[4+16*line + 12], rsp_data[4+16*line + 13], rsp_data[4+16*line + 14], rsp_data[4+16*line + 15]))
        # #print(read_mv,sum(bitarray))
    else:
        # repeat call
        if current_iter > max_repeats:
            print(f"Unable to read after {max_repeats} attempts. Skipping.") 
            return 0 if bit_mode else np.full((512, 16),2,dtype=np.uint8)
        else:
            print(f"Repeating call to read data at address {address}. Attempt #{current_iter} failed.")
            return handle_read_data(port, address,read_mv,vt,bit_mode, max_repeats=max_repeats, current_iter=current_iter+1)
    if bit_mode:
        return sum(bitarray)
    else:
        return NDarray

def handle_write_data(port, args):
    print("! Operation not supported")
    return False

def handle_ana_get_cal_counts(port, args):
    cmd_len = 8
    rsp_len = 16
    cmd_data = make_cmd(cmd_len, 80)
    insert_crc(cmd_data)
    port.write(cmd_data)

    rsp_data = read_rsp(port, rsp_len, 81)
    print(port.in_waiting)
    print(rsp_data,rsp_data[4:-4].decode('ascii'),len(rsp_data[4:-4]),len(rsp_data.decode('utf-16')))
    if len(rsp_data) >= rsp_len:
        ce_10v_cts, reset_10v_cts, wp_acc_10v_cts, spare_10v_cts = struct.unpack_from(
            "<HHHH", rsp_data, 4)
        print("  CE 10V Counts:       {}\r\n  Reset 10V Counts:    {}\r\n  WP/Acc 10V Counts:   {}\r\n  Spare 10V Counts:    {}".format(
            ce_10v_cts, reset_10v_cts, wp_acc_10v_cts, spare_10v_cts))
    else:
        return False
    return True

def handle_ana_set_cal_counts(port, args):
    ce_10v_cts = args.ce_cal_cts[0]
    reset_10v_cts = args.reset_cal_cts[0]
    wp_acc_10v_cts = args.wp_acc_cal_cts[0]
    spare_10v_cts = args.spare_cal_cts[0]
    cmd_len = 16
    rsp_len = 8
    cmd_data = make_cmd(cmd_len, 82)
    struct.pack_into("<HHHH", cmd_data, 4, ce_10v_cts, reset_10v_cts, wp_acc_10v_cts, spare_10v_cts)
    insert_crc(cmd_data)

    port.write(cmd_data)
    rsp_data=read_rsp(port, rsp_len, 83)
    if len(rsp_data) >= rsp_len:
        print("  Set cal counts")
    else:
        return False
    return True

analog_unit_map = {
    "ce" : 0,
    "reset" : 1,
    "wp_acc" : 2,
    "spare" : 3
}

def handle_ana_set_active_counts(port, args):
    unit = analog_unit_map.get(args.analog_unit[0])
    unit_counts = args.counts[0]
    cmd_len = 16
    rsp_len = 8
    cmd_data = make_cmd(cmd_len, 84)
    struct.pack_into("<II", cmd_data, 4, unit, unit_counts)
    insert_crc(cmd_data)

    port.write(cmd_data)
    rsp_data = read_rsp(port, rsp_len, 85)
    if len(rsp_data) >= rsp_len:
        print("  Set {} counts to {}".format(args.analog_unit[0], unit_counts))
    else:
        return False
    return True


def hex_int(x):
    return int(x, 16)


def get_dac_code(mv):
    gain = 10/1627*4095/3.3
    return int(mv/gain*4095/3.3)

def get_mv(dac_code):
    gain = 10/1627*4095/3.3
    return float(gain*3.3/4095*dac_code)

def read_chip_voltages(port,voltages,start_address=0,location='.'):
    count = 0
    printProgressBar(0, len(voltages)*128*1024, prefix='Getting sweep data: ', suffix='complete', decimals=0, length=50)
    for idx2, base_address in enumerate(range(start_address,67108864,65024 + 512)):
        for idx1, j in enumerate(voltages):
            saved_array = np.zeros((512,16))
            for idx, address in enumerate(range(base_address, base_address + 65024 + 512, 512)):
                NDarray = handle_read_data(port, address, j, 1, 0)
                if idx == 0:
                    saved_array = NDarray
                else:
                    saved_array = np.vstack((saved_array,NDarray))
                    #print(saved_array)
                #sg.one_line_progress_meter('Getting Sweep Data', count + 1, len(voltages) * 1024 *128)
                time.sleep(0.003) # This doesn't end up slowing us down much, but it does avoid costly serial timeout errors.
                printProgressBar(count, len(voltages)*128*1024, prefix='Getting sweep data: ', suffix='complete', decimals=0, length=50)
                count+=1
            np.savetxt(os.path.join(location,"data-{}-{}.csv".format(j,base_address)), saved_array, fmt='%d', delimiter='')

    return

def get_voltage_sweep(port,base_address,voltages,method):
    if not method:
        values = []
        tracker = 0
        printProgressBar(0, len(voltages)*128, prefix='Getting sweep data: ', suffix='complete', decimals=0, length=50)
        for idx1,j in enumerate(voltages):
            temp = []
            for idx, address in enumerate(range(base_address, base_address + 65024 + 512, 512)):
                NDarray = handle_read_data(port, address, j,1,1)
                if not isinstance(NDarray,bool):
                    temp.append(NDarray)
                #sg.one_line_progress_meter('Getting Sweep Data', tracker + 1, len(voltages)*128)
                printProgressBar(tracker+1, len(voltages)*128, prefix='Getting sweep data: ', suffix='complete', decimals=0, length=50)
                tracker+=1
            values.append([j, sum(temp)])

        v = [x[0] for x in values]
        bits = [x[1] for x in values]
    else:
        bits = []
        v = voltages
        printProgressBar(0, len(voltages), prefix='Getting sweep data: ', suffix='complete', decimals=0, length=50)
        for idx1, j in enumerate(voltages):
            bit_count = handle_get_sector_bit_count(port,base_address,read_mv=j)
            bits.append(bit_count)
            time.sleep(.003)
            printProgressBar(idx, len(voltages), prefix='Getting sweep data: ', suffix='complete', decimals=0, length=50)
            #sg.one_line_progress_meter('Getting Sweep Data', idx1 + 1, len(voltages))
    if plt.fignum_exists(1):
        ax1 = plt.figure(1).axes[0]
        ax2 = plt.figure(1).axes[1]
    else:
        fig1 = plt.figure(1)
        ax1 = fig1.gca()
        ax1.set_xlabel('Voltage (mV)')
        ax1.set_ylabel('Counts')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Sector Dist.')
    #print("v: ")
    #print(v)
    #print("bits: ")
    #print(bits)
    ax1.plot(v, bits, label='Sector: {}'.format(base_address))
    ax2.plot(v[1:], np.diff(bits), '--', label='Sector Dist.: {}, {}'.format(base_address, voltages[2] - voltages[1]))
    ax2.legend(loc='best')
    ax1.legend(loc='best')
    plt.show()
    return
def get_bit_noise(port, sector_address, voltages,iterations):
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
            NDarray = handle_read_data(port, sector_address, j,1,0)
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
            sg.one_line_progress_meter('Getting Noise Data', bar + 1, len(voltages)*iterations)
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

# CLI spec
parser = ArgumentParser(description="CLI for chip reads/writes")

parser.add_argument('-lp', '--list-ports', action='store_true', help='list USB port options')

parser.add_argument('-p', '--port', help='the USB port where the chip reader is installed')

parser.add_argument('-r',  '--read', action='store_true', help='read data from the chip')
parser.add_argument('--sector', type=int, help='the sector of the chip to read')
parser.add_argument('--start', type=int, help='the lowest voltage at which to read the chip in mV')
parser.add_argument('--stop', type=int, help='the highest voltage at which to read the chip in mV')
parser.add_argument('--step', type=int, help='the granularity in mV')
parser.add_argument('-a', '--all-sectors', action='store_true', help='read entire chip rather than a single sector')
parser.add_argument('--start-address', type=int, help='the lowest sector of the chip to read')
parser.add_argument('-d', '--directory', help='folder to contain output data files (relative path)')

parser.add_argument('-e', "--erase", action='store_true', help='erase data from the chip (always do this before writing)')

parser.add_argument('-w', '--write', action='store_true', help='write data to the chip')
parser.add_argument('-v', '--value', type=partial(int, base=0), help='the hex value to store in each byte')

args = parser.parse_args()


# Some nice argument handling to flag problems
if args.sector and args.all_sectors:
    parser.error("--sector and --all-sectors are incompatible")

if (args.read or args.write or args.erase) and not args.port:
    parser.error("--read, --write, and --erase always require --port \n"
                 "use --list-ports to see a list of available usb ports")
    
if args.read and (not (args.sector or args.all_sectors) or args.start is None or args.stop is None or args.step is None):
    parser.error("--read requires --sector or all-sectors, --start, --stop, and --step \n"
                 "suggested test values are --sector 800000 --start 2000 --stop 7000 --step 1000")

if args.write and (not (args.sector or args.all_sectors) or args.value is None):
    parser.error("--write requires --value and --sector or --all-sectors \n"
                 "suggested test values are --sector 800000 --value 0x5555")

if args.erase and not (args.sector or args.all_sectors):
    parser.error("--erase requires --sector or --all-sectors \n"
                 "suggested test value is --sector 800000")

if args.start_address and not (args.all_sectors and args.read):
    parser.error("--start-address may only be used in conjunction with --read and --all-sectors")


# Check if a port is valid and assign serial object if possible
if(args.port):
    print("Checking status of port " + args.port)
    try:
        ser = serial.Serial(args.port, 115200, bytesize = serial.EIGHTBITS,stopbits =serial.STOPBITS_ONE, parity  = serial.PARITY_NONE,timeout=2)
    except serial.SerialException as e:
        print(e)
        parser.error("It appears there was a problem with your USB port")

    resp = handle_ping(ser, args='')
        
    if not isinstance(resp,bool):
        print(f"USB {resp[1]}")
        print(f"USB status idle={bool(resp[2])}")
    else:
        print("USB ping failed")

    # Wait just a bit before the next call
    time.sleep(.01)
# List ports    
if(args.list_ports):
    ports = serial.tools.list_ports.comports()
    port_names = []
    for port, desc, hwid in sorted(ports):
        port_names.append(port)
    for name in port_names:
        print(name)

# Read a sector or entire chip
elif(args.read):
    if(args.sector):
        get_voltage_sweep(ser, args.sector, range(args.start, args.stop, args.step), 0)
    elif(args.all_sectors):
        if(args.directory):
            os.mkdir(args.directory)
        directory = args.directory if args.directory else '.'
        start_address = args.start_address if args.start_address else 0
        read_chip_voltages(ser, range(args.start, args.stop, args.step), start_address=start_address, location=directory)
        
# Erase a sector or entire chip
elif(args.erase):
    if(args.sector):
        handle_erase_sector(ser, args.sector)
    elif(args.all_sectors):
        handle_erase_chip(ser)

# Write a sector or entire chip
elif(args.write):
    if(args.sector):
        handle_program_sector(ser, args.sector, args.value)
    elif(args.all_sectors):
        handle_program_chip(ser, args.value)
        print("Writing a chip takes about 30 minutes and will continue despite the serial error that causes this macro to abort. Please wait until the chip stops flashing.")

    
#ports = serial.tools.list_ports.comports()
#port_names = []
#for port, desc, hwid in sorted(ports):
#    port_names.append(port)
#
#tab1_layout =  [[sg.Listbox(values=port_names, size=(30, 3), key='-PORTS-', enable_events=True)],
#                [sg.Text('Version', size=(15, 1)), sg.Text('', size=(15, 1),key='-VERSION-')],
#                [sg.Text('Idle?', size=(15, 1)), sg.Text('', size=(15, 1),key='-IDLE-')]
#                ]
#tab3_layout =  [[sg.Text('Sector Address', size=(15, 1)), sg.InputText('800000',key = '-SECTOR ADDRESS PROGRAM-')],
#                [sg.Text('Value', size=(15, 1)), sg.InputText('0x5555', size=(15, 1),key='-SECTOR VALUE-')],
#                [sg.Button('Program Chip', size=(15, 1), key = '-PROGRAM CHIP-'),sg.Button('Erase Chip', size=(15, 1), key = '-ERASE C#HIP-')],
#                [sg.Button('Program Sector', size=(15, 1), key = '-PROGRAM SECTOR-'),sg.Button('Erase Sector', size=(15, 1), key = '-E#RASE SECTOR-')]
#                ]
#tab2_layout = [
#    [sg.Text('Sector Address', size=(15, 1)), sg.InputText('800000',key = '-SECTOR ADDRESS-')],
#    [sg.Text('Sweep Start (mv)', size=(15, 1)), sg.InputText('2000',key = '-START MV-')],
#    [sg.Text('Sweep End (mv)', size=(15, 1)), sg.InputText('4000',key = '-END MV-')],
#    [sg.Text('Step (mV)', size=(15, 1)), sg.InputText('200',key = '-STEP MV-')],
#    [sg.Text('Iterations (#)', size=(15, 1)), sg.InputText('5',key = '-ITERS NOISE-')],
#    [sg.Checkbox('Calculation on Chip (1) vs on PC (0)', default=True,key = '-METHOD-')],
#    [sg.Button('Sweep Voltages', key = '-SWEEP SUBMIT-'),sg.Button('Bit Noise', key = '-BIT NOISE-'),sg.Button('Sweep Chip', key = '-S#WEEP CHIP SUBMIT-')]
#]
#layout = [[sg.TabGroup([[sg.Tab('Board Config', tab1_layout, tooltip='tip'),sg.Tab('Board Program', tab3_layout, tooltip='tip'), sg.Ta#b('Voltage Sweep', tab2_layout)]], tooltip='TIP2')],
#          [sg.Button('Exit')]]
#window = sg.Window('Neutron GUI', layout, default_element_size=(15,1))
#plt.ion()
#while True:
#    event, values = window.read()
#    if event == sg.WIN_CLOSED or event == 'Exit':
#        try:
#            ser.close()
#        except:
#            pass
#        break
#    if event == '-PORTS-':
#        print(values['-PORTS-'][0])
#        try:
#            ser = serial.Serial(values['-PORTS-'][0], 115200, bytesize = serial.EIGHTBITS,stopbits =serial.STOPBITS_ONE, parity  = ser#ial.PARITY_NONE,timeout=10)
#        except serial.SerialException as e:
#            print(e)
#            window['-VERSION-'].update('N/A')
#            window['-IDLE-'].update('N/A')
#
#            continue
#        resp = handle_ping(ser, args='')
#
#        #handle_ana_get_cal_counts(ser, args='')
#        if not isinstance(resp,bool):
#            window['-VERSION-'].update(resp[1])
#            window['-IDLE-'].update(resp[2])
#        else:
#            window['-VERSION-'].update('N/A')
#            window['-IDLE-'].update('N/A')
#    elif event == '-SWEEP SUBMIT-':
#        resp_values = [values['-SECTOR ADDRESS-'],values['-START MV-'],values['-END MV-'],values['-STEP MV-'],values['-METHOD-']]
#        try:
#            resp_values = [int(x) for x in resp_values]
#        except e:
#            print(e)
#            continue
#        get_voltage_sweep(ser, resp_values[0],range(resp_values[1],resp_values[2],resp_values[3]),resp_values[4])
#    elif event == '-ERASE SECTOR-':
#        handle_erase_sector(ser,int(values['-SECTOR ADDRESS PROGRAM-']))
#    elif event == '-ERASE CHIP-':
#        handle_erase_chip(ser)
#    elif event == '-PROGRAM SECTOR-':
#        handle_program_sector(ser,int(values['-SECTOR ADDRESS PROGRAM-']), int(values['-SECTOR VALUE-'],16))
#    elif event == '-BIT NOISE-':
#        resp_values = [values['-SECTOR ADDRESS-'], values['-START MV-'], values['-END MV-'], values['-STEP MV-'],values['-ITERS NOISE-#']]
#        try:
#            resp_values = [int(x) for x in resp_values]
#        except e:
#            print(e)
#            continue
#        get_bit_noise(ser, resp[0], range(resp_values[1],resp_values[2],resp_values[3]),resp_values[4])
#    elif event == '-PROGRAM CHIP-':
#        handle_program_chip(ser, int(values['-SECTOR VALUE-'],16))
#    elif event == '-SWEEP CHIP SUBMIT-':
#        resp_values = [values['-START MV-'], values['-END MV-'], values['-STEP MV-']]
#        try:
#            resp_values = [int(x) for x in resp_values]
#        except e:
#            print(e)
#            continue
#        read_chip_voltages(ser, range(resp_values[0], resp_values[1], resp_values[2]))
#window.close()

