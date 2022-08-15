"""Mock module for nisoc_readout"""

# Exceptions

class MessageValidationError(Exception):
    """Raised when there is an error validating a message"""
    pass

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

class Readout:
    readfunc = None
    writefunc = None

    def __init__(self, readfunc=None, writefunc=None):
        self.readfunc = readfunc
        self.writefunc = writefunc

    def ping(self) -> tuple[int, str, int]:
        return 1234, "test", 0

    def vt_get_bit_count_kpage(self, base_address: int, read_mv: int) -> list:
        return [0x5A] * 1024

    def erase_chip(self) -> None:
        pass

    def erase_sector(self, sector_address: int) -> None:
        pass

    def program_sector(self, sector_address: int, prog_value: int) -> None:
        pass

    def program_chip(self, prog_value: int) -> None:
        pass

    def get_sector_bit_count(self, base_address: int, read_mv: int) -> int:
        return 12340

    def read_data(self, base_address, vt_mode: bool=False, read_mv: int=0) -> bytearray:
        data = bytearray(1024)
        for i in range(0, 1024, 2):
            data[i] = 0x50
        for i in range(1, 1025, 2):
            data[i] = 0xFA
        return data

    def write_data(self, base_address: int, words) -> None:
        pass

    def ana_get_cal_counts(self) -> tuple[int, int, int, int]:
        return 16000, 16000, 16000, 16000

    def ana_set_cal_counts(self, ce_10v_cts: int, reset_10v_cts: int, wp_acc_10v_cts: int, spare_10v_cts: int) -> None:
        pass

    analog_unit_map = {
        "ce" : 0,
        "reset" : 1,
        "wp_acc" : 2,
        "spare" : 3
    }
    
    def ana_set_active_counts(self, unit: int, unit_counts: int) -> None:
        pass
