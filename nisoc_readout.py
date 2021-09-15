import crcmod
import struct

# CRC

crc_poly = 0x104C11DB7
crc_seed = 0x100000007
xor_out = 0

# CRCs are compatible with Renesas Synergy CRC hardware used the dumb way
# for CRC32-C: seed=0, xor_out=0, poly=0x11EDC6F41, reverse=false, reverse order of each 4-byte chunk
# for CRC32: seed=(seed), xor_out=0, poly=0x104C11DB7, reverse=false, reverse order of each 4-byte chunk

crc_f = crcmod.mkCrcFun(crc_poly, rev=False, initCrc=crc_seed, xorOut=xor_out)

def crc_4(arr):
    crc = crc_seed
    i = 0
    while i < len(arr):
        crc = crc_f(bytearray((arr[i+3], arr[i+2], arr[i+1], arr[i])), crc)
        i += 4
    return crc

# Exceptions

class MessageValidationError(Exception):
    """Raised when there is an error validating a message"""
    pass

class SerialTimeoutError(Exception):
    """Raised on timeout when reading serial port"""
    pass

# Communications

hdr_start_char = 0x7E

msg_ids = {
    0 : "null",
    2 : "cmd_ping",
    3 : "rsp_ping",
    4 : "rsp_unknown_cmd",
    5 : "rsp_failed",
    6 : "cmd_vt_get_bit_count_kpage",
    7 : "rsp_vt_get_bit_count_kpage",
    8 : "cmd_erase_chip",
    9 : "rsp_erase_chip",
    10 : "cmd_erase_sector",
    11 : "rsp_erase_sector",
    12 : "cmd_program_sector",
    13 : "rsp_program_sector",
    14 : "cmd_program_chip",
    15 : "rsp_program_chip",
    16 : "cmd_get_sector_bit_count",
    17 : "rsp_get_sector_bit_count",
    18 : "cmd_read_data",
    19 : "rsp_read_data",
    20 : "cmd_write_data",
    21 : "rsp_write_data",

    80 : "cmd_ana_get_cal_counts",
    81 : "rsp_ana_get_cal_counts",
    82 : "cmd_ana_set_cal_counts",
    83 : "rsp_ana_set_cal_counts",
    84 : "cmd_ana_set_active_counts",
    85 : "rsp_ana_set_active_counts"
}

def validate_msg(header, data, exp_id):
    start_char, length, id = struct.unpack_from("<BHB", header, 0)

    if start_char != 0x7E:
        raise MessageValidationError("Wrong start char {}".format(hex(start_char)))
    if len(data) < (length-4):
        raise MessageValidationError("Data length too short ({} < {})".format(len(data), length-4))
    if id != exp_id:
        raise MessageValidationError("Expected message ID {}. Got Message ID {} ({})".format(exp_id, id, msg_ids[id] if id in msg_ids else "unknown id"))

    msg_crc, = struct.unpack_from("<I", data, length-4-4)
    crc = crc_4(header)
    crc = crc_4(data[0:length-4-4], crc)

    if msg_crc != crc:
        raise MessageValidationError("msg crc {} != calc crc {}".format(hex(msg_crc), hex(crc)))
    return True

def make_cmd(cmd_len, id):
    data = bytearray(cmd_len)
    struct.pack_into("<BHB", data, 0, hdr_start_char, cmd_len, id)
    return data

def insert_crc(data):
    crc = crc_4(data[0:len(data)-4])
    struct.pack_into("<I", data, len(data)-4, crc)

def read_rsp(port, rsp_len, rsp_id):
    rsp_header = port.read(4)
    if len(rsp_header) < 4:
        raise SerialTimeoutError("Timeout reading message header. Expected 4 bytes, got {}".format(len(rsp_header)))

    msg_len, = struct.unpack_from("<H", rsp_header, 1)
    rsp_data = port.read(msg_len - 4)
    if len(rsp_data) < (msg_len-4):
        raise SerialTimeoutError("Timeout reading message payload from serial port. Expected {} bytes, got {}".format(msg_len-4, len(rsp_data)))

    validate_msg(rsp_header, rsp_data, rsp_id, True)
    return rsp_data

def ping(port):
    cmd_len = 8
    rsp_len = 32
    cmd_data = make_cmd(cmd_len, 2)
    insert_crc(cmd_data)

    port.write(cmd_data)
    rsp_data = read_rsp(port, rsp_len, 3)
    uptime, version_bin, is_busy = struct.unpack_from("<I16sI", rsp_data, 4)
    version = version_bin.decode('utf-8', 'ignore')

    return uptime, version, is_busy

def vt_get_bit_count_kpage(port, base_address, read_mv):
    cmd_len = 16
    rsp_len = 1032
    cmd_data = make_cmd(cmd_len, 6)
    struct.pack_into("<II", cmd_data, 4, base_address, read_mv)
    insert_crc(cmd_data)

    port.write(cmd_data)
    rsp_data = read_rsp(port, rsp_len, 7)

    return [rsp_data[4+i] for i in range(1024)]

def erase_chip(port):
    cmd_len = 8
    rsp_len = 8
    cmd_data = make_cmd(cmd_len, 8)
    insert_crc(cmd_data)

    port.write(cmd_data)
    rsp_data = read_rsp(port, rsp_len, 9)

def erase_sector(port, sector_address):
    cmd_len = 12
    rsp_len = 8
    cmd_data = make_cmd(cmd_len, 10)
    struct.pack_into("<I", cmd_data, 4, sector_address)
    insert_crc(cmd_data)

    port.write(cmd_data)
    rsp_data = read_rsp(port, rsp_len, 11)

def program_sector(port, sector_address, prog_value):
    cmd_len = 16
    rsp_len = 8
    cmd_data = make_cmd(cmd_len, 12)
    struct.pack_into("<IHH", cmd_data, 4, sector_address, prog_value, 0)
    insert_crc(cmd_data)

    port.write(cmd_data)
    rsp_data = read_rsp(port, rsp_len, 13)

def program_chip(port, prog_value):
    cmd_len = 12
    rsp_len = 8
    cmd_data = make_cmd(cmd_len, 14)
    struct.pack_into("<HH", cmd_data, 4, prog_value, 0)
    insert_crc(cmd_data)

    port.write(cmd_data)
    rsp_data = read_rsp(port, rsp_len, 15)

def get_sector_bit_count(port, base_address, read_mv):
    cmd_len = 16
    rsp_len = 12
    cmd_data = make_cmd(cmd_len, 16)
    struct.pack_into("<II", cmd_data, 4, base_address, read_mv)
    insert_crc(cmd_data)

    port.write(cmd_data)
    rsp_data = read_rsp(port, rsp_len, 17)
    bits_set, = struct.unpack_from("<I", rsp_data, 4)

    return bits_set

def read_data(port, base_address, vt_mode=False, read_mv=0):
    cmd_len = 20
    rsp_len = 1032
    cmd_data = make_cmd(cmd_len, 18)
    struct.pack_into("<III", cmd_data, 4, base_address, 1 if vt_mode else 0, read_mv)
    insert_crc(cmd_data)

    port.write(cmd_data)
    rsp_data = read_rsp(port, rsp_len, 19)

    data = []
    for i in range(0, 1024, 2):
        word, = struct.unpack_from("<H", rsp_data, 4 + 2*i)
        data.append(word)

    return data

def write_data(port, base_address, words):
    cmd_len = 1040
    rsp_len = 8
    cmd_data = make_cmd(cmd_len, 20)

    for i in range(len(words)):
        struct.pack_into("<H", cmd_data, 4, words[i])
    struct.pack_into("<II", cmd_data, 516, base_address, len(words))
    insert_crc(cmd_data)

    port.write(cmd_data)
    rsp_data = read_rsp(port, rsp_len, 21)

def ana_get_cal_counts(port):
    cmd_len = 8
    rsp_len = 16
    cmd_data = make_cmd(cmd_len, 80)
    insert_crc(cmd_data)

    port.write(cmd_data)
    rsp_data = read_rsp(port, rsp_len, 81)
    ce_10v_cts, reset_10v_cts, wp_acc_10v_cts, spare_10v_cts = struct.unpack_from("<HHHH", rsp_data, 4)

    return ce_10v_cts, reset_10v_cts, wp_acc_10v_cts, spare_10v_cts

def ana_set_cal_counts(port, ce_10v_cts, reset_10v_cts, wp_acc_10v_cts, spare_10v_cts):
    cmd_len = 16
    rsp_len = 8
    cmd_data = make_cmd(cmd_len, 82)
    struct.pack_into("<HHHH", cmd_data, 4, ce_10v_cts, reset_10v_cts, wp_acc_10v_cts, spare_10v_cts)
    insert_crc(cmd_data)

    port.write(cmd_data)
    rsp_data=read_rsp(port, rsp_len, 83)

analog_unit_map = {
    "ce" : 0,
    "reset" : 1,
    "wp_acc" : 2,
    "spare" : 3
}

def ana_set_active_counts(port, unit, unit_counts):
    # Analog units
    # CE#      0
    # RESET#   1
    # WP/ACC#  2
    # SPARE    3
    cmd_len = 16
    rsp_len = 8
    cmd_data = make_cmd(cmd_len, 84)
    struct.pack_into("<II", cmd_data, 4, unit, unit_counts)
    insert_crc(cmd_data)

    port.write(cmd_data)
    rsp_data = read_rsp(port, rsp_len, 85)
