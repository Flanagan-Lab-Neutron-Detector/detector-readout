import sys
import crcmod
import serial
import serial.tools.list_ports
import struct
import matplotlib.pyplot as plt
import numpy as np
import time
##import scipy
##import PySimpleGUI as sg      

##layout = [
##    [sg.Text('Enter Data')],
##    [sg.Text('Sector', size=(15, 1)), sg.InputText('800000')],
##    [sg.Text('Address', size=(15, 1)), sg.InputText()],
##    [sg.Text('Phone', size=(15, 1)), sg.InputText()],
##    [sg.Submit(), sg.Cancel()]
##]     
##
##window = sg.Window('Window Title', layout)    
##
##event, values = window.read()    
##window.close()
##
##text_input = values['-IN-']    
##sg.popup('You entered', text_input)

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
##    rsp_data = port.read(rsp_len)
    rsp_data = port.readline()
    if len(rsp_data) >= rsp_len:
        if msg_is_valid(rsp_data, rsp_id):
            return rsp_data
        else:
            print("! Invalid response {}".format(rsp_data))
            return bytearray()
    else:
        print("! Error: Serial timeout")
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
        print("ping rsp")
        print("  Uptime:               {}\r\n  Version String:       {}\r\n  Is busy:              {}".format(uptime, version, is_busy))
    else:
        return False
    return True

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

def handle_erase_chip(port, args):
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

def handle_erase_sector(sector_address):
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
    return True

def handle_program_chip(port, args):
    prog_value = args.program_value[0]
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
        print("Sector: {}\nVoltage: {}mV\nCounts: {}\n".format(sector_address, read_mv, bits_set))
    else:
        return False
    return bits_set

def handle_read_data(port, base_address,read_mv,vt):
    vt_mode = 1 if vt is not None else 0
    read_mv = read_mv if vt_mode else 0

    cmd_len = 20
    rsp_len = 1032
    cmd_data = make_cmd(cmd_len, 18)
    struct.pack_into("<III", cmd_data, 4, base_address, vt_mode, read_mv)
    insert_crc(cmd_data)

    port.write(cmd_data)
    rsp_id = 19
    rsp_data = read_rsp(port, rsp_len, rsp_id)
    #if not rsp_data:
    #   rsp_data = read_rsp(port, 1024, rsp_id)
        
    if len(rsp_data) >= rsp_len:
        array = np.frombuffer(rsp_data[4:-4], dtype=np.uint16)  # or dtype=np.dtype('<f4')
        bitarray = np.unpackbits(array.view(np.uint8))

        NDarray = np.reshape(bitarray, (512, 16)) #if we are reading 16 bit words
        for line in range(32):
             print("  {:08X}    {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X}".format(base_address + 16*line,
                                                                                                                                                        rsp_data[4+16*line + 0], rsp_data[4+16*line + 1], rsp_data[4+16*line + 2], rsp_data[4+16*line + 3],
                                                                                                                                                        rsp_data[4+16*line + 4], rsp_data[4+16*line + 5], rsp_data[4+16*line + 6], rsp_data[4+16*line + 7],
                                                                                                                                                        rsp_data[4+16*line + 8], rsp_data[4+16*line + 8], rsp_data[4+16*line + 8], rsp_data[4+16*line + 9],
                                                                                                                                                        rsp_data[4+16*line + 12], rsp_data[4+16*line + 13], rsp_data[4+16*line + 14], rsp_data[4+16*line + 15]))
        print(read_mv,sum(bitarray))
    else:
        return False
    return sum(bitarray)#NDarray

def handle_write_data(port, args):
    print("! Operation not supported")
    return False

def handle_ana_get_cal_counts(port, args):
    cmd_len = 8
    rsp_len = 16
    cmd_data = make_cmd(cmd_len, 80)
    insert_crc(cmd_data)
    print(cmd_data)
    port.write(cmd_data)

    rsp_data = read_rsp(port, rsp_len, 81)
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

# open serial port
try:
    ports = serial.tools.list_ports.comports()
    for port, desc, hwid in sorted(ports):
        print("{}: {} [{}]".format(port, desc, hwid))
    port = '/dev/ttyUSB0'
    port = serial.Serial(port, 115200, timeout=10)
    if not port.is_open:
        port.open()
    handle_ping(port,args = '')
    handle_ana_get_cal_counts(port,args = '')
    sector_address = 300000
    handle_erase_sector(sector_address)

    handle_program_sector(port, sector_address, 0xAAAA)
    values = []
    voltages = range(0,7000,500)
    for i in voltages:
        value = handle_read_data(port, sector_address, i, 0)
        if isinstance(value,int):
        	values.append([i,value])
    v = [x[0] for x in values]
    bits = [x[1] for x in values]
    plt.plot(v,bits)
    #plt.plot(voltages[1:],np.diff(values))
    
    plt.show()
    # handle_erase_sector(sector_address)
    # #handle_read_data(port, sector_address, 2750, 1)
    # avg_array = []
    # max = 2
    # real_count = 0
    #
    # results_matrix = []
    # map = {}
    # for i in range(max):
    #     data_matrix = []
    #     for idx,j in enumerate(range(2500,3500,10)):
    #         start_time = time.time()
    #
    #         #print(idx,i,j)
    #         NDarray = handle_read_data(port, sector_address, j,1)
    #         if not isinstance(NDarray,bool):
    #             data_matrix.append([j,NDarray])
    #             # vgrad = np.gradient(NDarray)
    #             # xgrad = vgrad[0]
    #             # x, y = range(0, xgrad.shape[0]), range(0, xgrad.shape[1])
    #             # #xi, yi = np.meshgrid(x, y)
    #             #rbf = scipy.interpolate.Rbf(xi, yi, xgrad)
    #             # hf = plt.figure()
    #             # ha = hf.add_subplot(111, projection='3d')
    #             # X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    #             # print(xgrad.shape,X.shape,Y.shape)
    #             # ha.plot_surface(X.T, Y.T, xgrad)
    #             # plt.show()
    #             if i == 0:
    #                 map[idx] = j
    #             current_time = time.time()
    #             elapsed_time = current_time - start_time
    #     results_matrix.append(data_matrix)
    # true_values = []
    # ix = 304
    # iy = 7
    #
    # for sweep in results_matrix:
    #     previous_value = -1
    #     true_value = 0
    #     # for ix, iy in np.ndindex(sweep[0][1].shape):
    #     for idx,[mv,matrix] in enumerate(sweep):
    #         value = matrix[ix,iy]*mv
    #         if previous_value == 0 and value >0:
    #             true_value = value
    #             true_values.append(true_value)
    #         previous_value = value
    #         print(ix,iy,true_value)
    # plt.hist(true_values)
    # print(np.mean(true_values),np.std(true_values))
    # plt.show()
    #plt.imshow(avg_array/real_count, cmap='hot', aspect='auto', interpolation='none', vmin=0, vmax=1, )
    #plt.colorbar()
    #plt.show()
except Exception as e:
    port.close()
    print(e)
    exit(1)
