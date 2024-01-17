"""Simulated readout for testing"""

import io
from os import SEEK_CUR
import struct
from datetime import datetime
from .messaging import HDR_START_CHAR, MSG_NAMES, MSG_IDS, make_cmd, insert_crc, crc_4

class ReadoutSimulator:
    reset_time: datetime
    reset_flags: int
    current_task: int
    current_task_state: int
    cfg_regs: dict[int, int]
    serial_buffer: io.BytesIO

    def __init__(self):
        self.serial_buffer = io.BytesIO()
        self.reset()

    def reset(self) -> None:
        """Reset simulator"""
        self.reset_time = datetime.now()
        self.reset_flags = 0
        self.current_task = 0
        self.current_task_state = 0

        self.cfg_regs = {
            0x0100 : 0x0001,
            0x0101 : 0x4F0E,
            0x0102 : 0x1115
        }

    def get_uptime(self) -> int:
        return int((datetime.now() - self.reset_time).total_seconds())

    def get_busy(self) -> bool:
        return False

    def handle_ping(self, payload: bytes) -> bytes:
        rsp_data = bytearray(36)
        struct.pack_into("<I16sIIII", rsp_data, 4, self.get_uptime(), b"readout sim\x00\x00\x00\x00\x00",
                         self.get_busy(), self.current_task, self.current_task_state)
        return rsp_data, MSG_IDS['rsp_ping']

    def handle_vt_get_bit_count_kpage(self, payload: bytes) -> bytes:
        rsp_data = bytearray(1024)
        for i in range(1024):
            rsp_data[4+i] = 0
        return rsp_data, MSG_IDS['rsp_vt_get_bit_count_kpage']

    def handle_erase_chip(self, payload: bytes) -> bytes:
        return bytes(), MSG_IDS['rsp_erase_chip']

    def handle_erase_sector(self, payload: bytes) -> bytes:
        return bytes(), MSG_IDS['rsp_erase_sector']

    def handle_program_sector(self, payload: bytes) -> bytes:
        return bytes(), MSG_IDS['rsp_program_sector']

    def handle_program_chip(self, payload: bytes) -> bytes:
        return bytes(), MSG_IDS['rsp_program_chip']

    def handle_get_sector_bit_count(self, payload: bytes) -> bytes:
        rsp_data = bytearray(4)
        struct.pack_into("<I", rsp_data, 0, 0)
        return rsp_data, MSG_IDS['rsp_get_sector_bit_count']

    def handle_read_data(self, payload: bytes) -> bytes:
        count, = struct.unpack_from("<I", payload, 12)
        rsp_data = bytearray(count*2)
        return rsp_data, MSG_IDS['rsp_read_data']

    def handle_write_data(self, payload: bytes) -> bytes:
        return bytes(), MSG_IDS['rsp_write_data']

    def handle_read_word(self, payload: bytes) -> bytes:
        rsp_data = bytearray(2)
        struct.pack_into("<H", rsp_data, 0, 0)
        return rsp_data, MSG_IDS['rsp_read_word']

    def handle_write_cfg(self, payload: bytes) -> bytes:
        address, data = struct.unpack_from("<II", payload, 0)
        self.cfg_regs[address] = data
        return bytes(), MSG_IDS['rsp_write_cfg']

    def handle_read_cfg(self, payload: bytes) -> bytes:
        address, = struct.unpack_from("<I", payload, 0)
        rsp_data = bytearray(4)
        struct.pack_into("<I", rsp_data, 0, self.cfg_regs[address] if address in self.cfg_regs else 0)
        return rsp_data, MSG_IDS['rsp_read_cfg']

    def handle_cfg_flash_enter(self, payload: bytes) -> bytes:
        return bytes(), MSG_IDS['rsp_cfg_flash_enter']

    def handle_cfg_flash_exit(self, payload: bytes) -> bytes:
        self.reset()
        return bytes(), MSG_IDS['rsp_cfg_flash_exit']

    def handle_cfg_flash_read(self, payload: bytes) -> bytes:
        address, count = struct.unpack_from("<II", payload, 0)
        rsp_data = bytearray(count)
        for i in range(count):
            rsp_data[i] = 0
        return rsp_data, MSG_IDS['rsp_cfg_flash_read']

    def handle_cfg_flash_write(self, payload: bytes) -> bytes:
        return bytes(), MSG_IDS['rsp_cfg_flash_write']

    def handle_cfg_flash_erase(self, payload: bytes) -> bytes:
        return bytes(), MSG_IDS['rsp_cfg_flash_erase']

    def handle_cfg_flash_dev_info(self, payload: bytes) -> bytes:
        rsp_data = bytearray(12)
        struct.pack_into("<BBBB4sBBBB", rsp_data, 0, 0, 1, 2, 3, b"9876", 0, 0, 0, 0)
        return rsp_data, MSG_IDS['rsp_cfg_flash_dev_info']

    def handle_ana_get_cal_counts(self, payload: bytes) -> bytes:
        rsp_data = bytearray(8)
        struct.pack_into("<ff", rsp_data, 0, 0.0, 1.0)
        return rsp_data, MSG_IDS['rsp_ana_get_cal_counts']

    def handle_ana_set_cal_counts(self, payload: bytes) -> bytes:
        return bytes(), MSG_IDS['rsp_ana_set_cal_counts']

    def handle_ana_set_active_counts(self, payload: bytes) -> bytes:
        return bytes(), MSG_IDS['rsp_ana_set_active_counts']

    def handle_unknown_cmd(self, payload: bytes) -> bytes:
        return bytes(), MSG_IDS['rsp_unknown_cmd']

    def process_message(self, id, payload) -> tuple[bytes, int]:
        if id == MSG_IDS['cmd_ping']:
            return self.handle_ping(payload)
        elif id == MSG_IDS['cmd_vt_get_bit_count_kpage']:
            return self.handle_vt_get_bit_count_kpage(payload)
        elif id == MSG_IDS['cmd_erase_chip']:
            return self.handle_erase_chip(payload)
        elif id == MSG_IDS['cmd_erase_sector']:
            return self.handle_erase_sector(payload)
        elif id == MSG_IDS['cmd_program_sector']:
            return self.handle_program_sector(payload)
        elif id == MSG_IDS['cmd_program_chip']:
            return self.handle_program_chip(payload)
        elif id == MSG_IDS['cmd_get_sector_bit_count']:
            return self.handle_get_sector_bit_count(payload)
        elif id == MSG_IDS['cmd_read_data']:
            return self.handle_read_data(payload)
        elif id == MSG_IDS['cmd_write_data']:
            return self.handle_write_data(payload)
        elif id == MSG_IDS['cmd_read_word']:
            return self.handle_read_word(payload)
        elif id == MSG_IDS['cmd_write_cfg']:
            return self.handle_write_cfg(payload)
        elif id == MSG_IDS['cmd_read_cfg']:
            return self.handle_read_cfg(payload)
        elif id == MSG_IDS['cmd_cfg_flash_enter']:
            return self.handle_cfg_flash_enter(payload)
        elif id == MSG_IDS['cmd_cfg_flash_exit']:
            return self.handle_cfg_flash_exit(payload)
        elif id == MSG_IDS['cmd_cfg_flash_read']:
            return self.handle_cfg_flash_read(payload)
        elif id == MSG_IDS['cmd_cfg_flash_write']:
            return self.handle_cfg_flash_write(payload)
        elif id == MSG_IDS['cmd_cfg_flash_erase']:
            return self.handle_cfg_flash_erase(payload)
        elif id == MSG_IDS['cmd_cfg_flash_dev_info']:
            return self.handle_cfg_flash_dev_info(payload)
        elif id == MSG_IDS['cmd_ana_get_cal_counts']:
            return self.handle_ana_get_cal_counts(payload)
        elif id == MSG_IDS['cmd_ana_set_cal_counts']:
            return self.handle_ana_set_cal_counts(payload)
        elif id == MSG_IDS['cmd_ana_set_active_counts']:
            return self.handle_ana_set_active_counts(payload)
        else:
            return self.handle_unknown_cmd(payload)

    def ser_read(self, n: int) -> bytes:
        #print(f"[ReadoutSimulator] Reading {n} bytes")
        return self.serial_buffer.read(n)

    def ser_write(self, data: bytes) -> None:
        # Validate incoming message
        if len(data) < 8:
            raise ValueError(f"Data length too short ({len(data)} < 8)")
        if data[0] != HDR_START_CHAR:
            raise ValueError(f"Wrong start char {data[0]:02X}h, expected {HDR_START_CHAR:02X}h")
        msg_len, = struct.unpack_from("<H", data, 1)
        if len(data) < msg_len:
            raise ValueError(f"Data length too short ({len(data)} < {msg_len})")
        if len(data) > msg_len:
            raise ValueError(f"Data length too long ({len(data)} > {msg_len})")
        msg_crc, = struct.unpack_from("<I", data, msg_len-4)
        crc = crc_4(data[0:msg_len-4])
        if msg_crc != crc:
            raise ValueError(f"Message crc {msg_crc:08X}h != calc crc {crc:08X}h")
        id, = struct.unpack_from("<B", data, 3)
        if id not in MSG_NAMES:
            raise ValueError(f"Unknown message ID {id}")

        #print(f"[ReadoutSimulator] Processing message {id} ({MSG_NAMES[id] if id in MSG_NAMES else 'unknown message'})")

        # Process the message
        rsp_payload, id = self.process_message(id, data[4:msg_len-4])
        rsp = make_cmd(len(rsp_payload)+8, id)
        rsp[4:len(rsp_payload)+4] = rsp_payload
        insert_crc(rsp)

        #print(f"[ReadoutSimulator] Writing message {id} ({MSG_NAMES[id] if id in MSG_NAMES else 'unknown message'}), {len(rsp)} bytes")

        self.serial_buffer.write(rsp)
        self.serial_buffer.seek(-len(rsp), SEEK_CUR)