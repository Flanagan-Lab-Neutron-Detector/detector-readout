class MessageValidationError(Exception):
    """Raised when there is an error validating a message"""
    pass

def ping(port) -> tuple[int, str, int]:
    return 1234, "test", 0

def vt_get_bit_count_kpage(port, base_address: int, read_mv: int) -> list:
    return [0x5A] * 1024

def erase_chip(port):
    pass

def erase_sector(port, sector_address: int):
    pass

def program_sector(port, sector_address: int, prog_value: int):
    pass

def program_chip(port, prog_value: int):
    pass

def get_sector_bit_count(port, base_address: int, read_mv: int) -> int:
    return 12340

def read_data(port, base_address, vt_mode: bool=False, read_mv: int=0) -> bytearray:
    data = bytearray(1024)
    for i in range(0, 1024, 2):
        data[i] = 0x50
    for i in range(1, 1025, 2):
        data[i] = 0xFA
    return data

def write_data(port, base_address: int, words):
    pass

def ana_get_cal_counts(port) -> tuple[int, int, int, int]:
    return 16000, 16000, 16000, 16000

def ana_set_cal_counts(port, ce_10v_cts: int, reset_10v_cts: int, wp_acc_10v_cts: int, spare_10v_cts: int):
    pass

analog_unit_map = {
    "ce" : 0,
    "reset" : 1,
    "wp_acc" : 2,
    "spare" : 3
}

def ana_set_active_counts(port, unit: int, unit_counts: int):
    pass
