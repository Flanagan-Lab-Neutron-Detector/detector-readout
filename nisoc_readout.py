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

def crc_4(arr, start=crc_seed):
    crc = start
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

msg_names = {
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

msg_ids = { msg_names[msg_id]: msg_id for msg_id in msg_names }


def validate_msg(header: bytearray, data: bytearray, exp_id: int) -> bool:
    start_char, length, id = struct.unpack_from("<BHB", header, 0)

    if start_char != 0x7E:
        raise MessageValidationError("Wrong start char {}".format(hex(start_char)))
    if len(data) < (length-4):
        raise MessageValidationError("Data length too short ({} < {})".format(len(data), length-4))
    if id != exp_id:
        raise MessageValidationError("Expected message ID {}. Got Message ID {} ({})".format(exp_id, id, msg_names[id] if id in msg_names else "unknown message"))

    msg_crc, = struct.unpack_from("<I", data, length-4-4)
    crc = crc_4(header)
    crc = crc_4(data[0:length-4-4], crc)

    if msg_crc != crc:
        raise MessageValidationError("Message crc {} != calc crc {} for message {} ({})".format(hex(msg_crc), hex(crc)), id, msg_names[id] if id in msg_names else "unknown message")
    return True

def make_cmd(cmd_len: int, id: int) -> bytearray:
    data = bytearray(cmd_len)
    struct.pack_into("<BHB", data, 0, hdr_start_char, cmd_len, id)
    return data

def insert_crc(data: bytearray) -> None:
    crc = crc_4(data[0:len(data)-4])
    struct.pack_into("<I", data, len(data)-4, crc)

def read_rsp(port, rsp_id: int) -> bytes:
    rsp_header = port.read(4)
    if len(rsp_header) < 4:
        raise SerialTimeoutError("Timeout reading message header. Expected 4 bytes, got {}".format(len(rsp_header)))

    msg_len, = struct.unpack_from("<H", rsp_header, 1)
    rsp_data = port.read(msg_len - 4)
    if len(rsp_data) < (msg_len-4):
        raise SerialTimeoutError("Timeout reading message payload from serial port. Expected {} bytes, got {}".format(msg_len-4, len(rsp_data)))

    validate_msg(rsp_header, rsp_data, rsp_id)
    return rsp_data

def ping(port) -> tuple[int, str, int]:
    cmd_len = 8
    #rsp_len = 32
    cmd_data = make_cmd(cmd_len, msg_ids['cmd_ping'])
    insert_crc(cmd_data)

    port.write(cmd_data)
    rsp_data = read_rsp(port, msg_ids['rsp_ping'])
    uptime, version_bin, is_busy = struct.unpack_from("<I16sI", rsp_data, 4)
    version = version_bin.decode('utf-8', 'ignore')

    return uptime, version, is_busy

def vt_get_bit_count_kpage(port, base_address: int, read_mv: int) -> list:
    cmd_len = 16
    #rsp_len = 1032
    cmd_data = make_cmd(cmd_len, msg_ids['cmd_vt_get_bit_count_kpage'])
    struct.pack_into("<II", cmd_data, 4, base_address, read_mv)
    insert_crc(cmd_data)

    port.write(cmd_data)
    rsp_data = read_rsp(port, msg_ids['rsp_vt_get_bit_count_kpage'])

    # TODO(aidan): Figure out what this is supposed to do
    return [rsp_data[4+i] for i in range(1024)]

def erase_chip(port) -> None:
    cmd_len = 8
    #rsp_len = 8
    cmd_data = make_cmd(cmd_len, msg_ids['cmd_erase_chip'])
    insert_crc(cmd_data)

    port.write(cmd_data)
    _ = read_rsp(port, msg_ids['rsp_erase_chip'])

def erase_sector(port, sector_address: int) -> None:
    cmd_len = 12
    #rsp_len = 8
    cmd_data = make_cmd(cmd_len, msg_ids['cmd_erase_sector'])
    struct.pack_into("<I", cmd_data, 4, sector_address)
    insert_crc(cmd_data)

    port.write(cmd_data)
    _ = read_rsp(port, msg_ids['rsp_erase_sector'])

def program_sector(port, sector_address: int, prog_value: int) -> None:
    cmd_len = 16
    #rsp_len = 8
    cmd_data = make_cmd(cmd_len, msg_ids['cmd_program_sector'])
    struct.pack_into("<IHH", cmd_data, 4, sector_address, prog_value, 0)
    insert_crc(cmd_data)

    port.write(cmd_data)
    _ = read_rsp(port, msg_ids['rsp_program_sector'])

def program_chip(port, prog_value: int) -> None:
    cmd_len = 12
    #rsp_len = 8
    cmd_data = make_cmd(cmd_len, msg_ids['cmd_program_chip'])
    struct.pack_into("<HH", cmd_data, 4, prog_value, 0)
    insert_crc(cmd_data)

    port.write(cmd_data)
    _ = read_rsp(port, msg_ids['rsp_program_chip'])

def get_sector_bit_count(port, base_address: int, read_mv: int) -> int:
    cmd_len = 16
    #rsp_len = 12
    cmd_data = make_cmd(cmd_len, msg_ids['cmd_get_sector_bit_count'])
    struct.pack_into("<II", cmd_data, 4, base_address, read_mv)
    insert_crc(cmd_data)

    port.write(cmd_data)
    rsp_data = read_rsp(port, msg_ids['rsp_get_sector_bit_count'])
    bits_set, = struct.unpack_from("<I", rsp_data, 4)

    return bits_set

def read_data(port, base_address, vt_mode: bool=False, read_mv: int=4000) -> bytearray:
    cmd_len = 20
    #rsp_len = 1032
    cmd_data = make_cmd(cmd_len, msg_ids['cmd_read_data'])
    struct.pack_into("<III", cmd_data, 4, base_address, 1 if vt_mode else 0, read_mv)
    insert_crc(cmd_data)

    port.write(cmd_data)
    rsp_data = read_rsp(port, msg_ids['rsp_read_data'])

    # data = []
    # for i in range(0, 1024, 2):
    #     word, = struct.unpack_from("<H", rsp_data, 4 + 2*i)
    #     data.append(word)

    return bytearray(rsp_data[4:-4])

def write_data(port, base_address: int, words: list | bytes | bytearray) -> None:
    cmd_len = 1040
    #rsp_len = 8
    cmd_data = make_cmd(cmd_len, msg_ids['cmd_write_data'])

    for i in range(len(words)):
        struct.pack_into("<H", cmd_data, 4, words[i])
    struct.pack_into("<II", cmd_data, 516, base_address, len(words))
    insert_crc(cmd_data)

    port.write(cmd_data)
    _ = read_rsp(port, msg_ids['rsp_write_data'])

def ana_get_cal_counts(port) -> tuple[int, int, int, int]:
    cmd_len = 8
    #rsp_len = 16
    cmd_data = make_cmd(cmd_len, msg_ids['cmd_ana_get_cal_counts'])
    insert_crc(cmd_data)

    port.write(cmd_data)
    rsp_data = read_rsp(port, msg_ids['rsp_ana_get_cal_counts'])
    ce_10v_cts, reset_10v_cts, wp_acc_10v_cts, spare_10v_cts = struct.unpack_from("<HHHH", rsp_data, 4)

    return ce_10v_cts, reset_10v_cts, wp_acc_10v_cts, spare_10v_cts

def ana_set_cal_counts(port, ce_10v_cts: int, reset_10v_cts: int, wp_acc_10v_cts: int, spare_10v_cts: int) -> None:
    cmd_len = 16
    #rsp_len = 8
    cmd_data = make_cmd(cmd_len, msg_ids['cmd_ana_set_cal_counts'])
    struct.pack_into("<HHHH", cmd_data, 4, ce_10v_cts, reset_10v_cts, wp_acc_10v_cts, spare_10v_cts)
    insert_crc(cmd_data)

    port.write(cmd_data)
    _ = read_rsp(port, msg_ids['rsp_ana_set_cal_counts'])

analog_unit_map = {
    "ce" : 0,
    "reset" : 1,
    "wp_acc" : 2,
    "spare" : 3
}

def ana_set_active_counts(port, unit: int, unit_counts: int) -> None:
    # Analog units
    # CE#      0
    # RESET#   1
    # WP/ACC#  2
    # SPARE    3
    cmd_len = 16
    #rsp_len = 8
    cmd_data = make_cmd(cmd_len, msg_ids['cmd_ana_set_active_counts'])
    struct.pack_into("<II", cmd_data, 4, unit, unit_counts)
    insert_crc(cmd_data)

    port.write(cmd_data)
    _ = read_rsp(port, msg_ids['rsp_ana_set_active_counts'])
