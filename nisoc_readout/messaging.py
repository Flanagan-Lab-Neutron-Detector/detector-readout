"""Messaging routines for nisoc_readout.

Contains routines for exchanging messages with NISoC readouts, including validation failure parsing.
"""

import crcmod
import struct
from collections.abc import Sequence
from typing import Callable

# CRC

CRC_POLY = 0x104C11DB7
CRC_SEED = 0x100000007
CRC_XOR_OUT = 0

# CRCs are compatible with Renesas Synergy CRC hardware used the default way
# for CRC32-C: seed=0, CRC_XOR_OUT=0, poly=0x11EDC6F41, reverse=false, reverse order of each 4-byte chunk
# for CRC32: seed=(seed), CRC_XOR_OUT=0, poly=0x104C11DB7, reverse=false, reverse order of each 4-byte chunk

# Calculate CRC for a collection of values. Signature is crc_f(vals, init_crc). The device reverses each 4-byte chunk.
crc_f = crcmod.mkCrcFun(CRC_POLY, rev=False, initCrc=CRC_SEED, xorOut=CRC_XOR_OUT)

def crc_4(arr: Sequence, start=CRC_SEED):
	"""Calculate CRC for elements in arr in 4-byte chunks, with initial value."""
	crc = start
	i = 0
	while i < len(arr):
		crc = crc_f(bytearray((arr[i+3], arr[i+2], arr[i+1], arr[i])), crc)
		i += 4
	return crc

# Exceptions

class MessageValidationError(Exception):
	"""Raised when there is an error validating a message"""

	def __init__(self, desc, msgheader: bytes, msgdata: bytes):
		super().__init__(desc)
		self.desc      = desc
		self.msgheader = msgheader
		self.msgdata   = msgdata

class MessageFailureError(MessageValidationError):
	"""Raised when a failure response is received"""

	def __init__(self, desc, msgheader: bytes, msgdata: bytes):
		super().__init__(desc, msgheader, msgdata)
		print(f"len(msgdata) = {len(msgdata)}")
		if len(msgdata) >= 4:
			self.reported_code_count, = struct.unpack_from("<I", msgdata, 0)
			sent_code_count = (len(msgdata) - 8) // 4
			self.failure_codes = [struct.unpack_from("<HH", msgdata, 4 + 4*code) for code in range(sent_code_count)]
			print(f"{self.reported_code_count} {sent_code_count} {self.failure_codes}")
			print(f"{msgdata}")
		else:
			self.reported_code_count = 0
			self.failure_codes = []

	def __str__(self):
		plural = lambda v: '' if v == 1 else 's'
		s = ''
		if self.reported_code_count == 0 and len(self.failure_codes) == 0:
			s = "Unknown failure (no codes reported)"
		elif len(self.failure_codes) == 0:
			failures = f"failure{plural(self.reported_code_count)}"
			s = f"Unknown {failures}: {self.reported_code_count} {failures} reported but none sent"
		elif self.reported_code_count == 0:
			failures = f"failure{plural(len(self.failure_codes))}"
			s = f"Unknown failures: No failures reported but additional data sent"
		elif self.reported_code_count > len(self.failure_codes):
			repfailures  = f"failure{plural(self.reported_code_count)}"
			sentfailures = f"failure{plural(len(self.failure_codes))}"
			s = f"Unknown failures: {self.reported_code_count} {repfailures} reported but only {len(self.failure_codes)} {sentfailures} sent: {self.failure_codes}"
		elif self.reported_code_count < len(self.failure_codes):
			s = f"{self.reported_code_count} failure{plural(self.reported_code_count)} (additional data sent): {self.failure_codes[0:self.reported_code_count]}"
		elif self.reported_code_count == len(self.failure_codes):
			s = f"{self.reported_code_count} failure{plural(self.reported_code_count)}: {self.failure_codes}"
		else:
			return super().__str__()
		return f"{super().__str__()}: {s}"


class SerialTimeoutError(Exception):
	"""Raised on timeout when reading serial port"""
	pass

# Communications

HDR_START_CHAR = 0x7E

MSG_NAMES = {
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
	22 : "cmd_read_word",
	23 : "rsp_read_word",
	24 : "cmd_write_cfg",
	25 : "rsp_write_cfg",
	26 : "cmd_read_cfg",
	27 : "rsp_read_cfg",

	28 : "cmd_cfg_flash_enter",
	29 : "rsp_cfg_flash_enter",
	30 : "cmd_cfg_flash_exit",
	31 : "rsp_cfg_flash_exit",
	32 : "cmd_cfg_flash_read",
	33 : "rsp_cfg_flash_read",
	34 : "cmd_cfg_flash_write",
	35 : "rsp_cfg_flash_write",
	36 : "cmd_cfg_flash_erase",
	37 : "rsp_cfg_flash_erase",
	38 : "cmd_cfg_flash_dev_info",
	39 : "rsp_cfg_flash_dev_info",

	80 : "cmd_ana_get_cal_counts",
	81 : "rsp_ana_get_cal_counts",
	82 : "cmd_ana_set_cal_counts",
	83 : "rsp_ana_set_cal_counts",
	84 : "cmd_ana_set_active_counts",
	85 : "rsp_ana_set_active_counts"
}

MSG_IDS = { MSG_NAMES[msg_id]: msg_id for msg_id in MSG_NAMES }

def validate_msg(header: bytearray, data: bytearray, exp_id: int) -> bool:
	# Validate start char and total length
	start_char, length, id = struct.unpack_from("<BHB", header, 0)
	if start_char != HDR_START_CHAR:
		raise MessageValidationError(f"Wrong start char {start_char:02X}h, expected {HDR_START_CHAR:02X}h", header, data)
	if len(data) < (length-4):
		raise MessageValidationError(f"Data length too short ({len(data)} < {length-4})", header, data)

	# Validate CRC
	msg_crc, = struct.unpack_from("<I", data, length-4-4)
	crc = crc_4(header)
	crc = crc_4(data[0:length-4-4], crc)
	if msg_crc != crc:
		raise MessageValidationError("Message crc {msg_crc:08X}h != calc crc {crc:08X}h for message {id} ({MSG_NAMES[id] if id in MSG_NAMES else 'unknown message')})", header, data)

	# check for failure or unexpected response
	if id == MSG_IDS["rsp_failed"]:
		raise MessageFailureError(f"Message failed: ID {exp_id} ({MSG_NAMES[exp_id] if exp_id in MSG_NAMES else 'unknown message'})", header, data)
	if id != exp_id:
		raise MessageValidationError(f"Expected message ID {exp_id}. Got Message ID {id} ({MSG_NAMES[id] if id in MSG_NAMES else 'unknown message'})", header, data)

	return True

def read_rsp(readfunc: Callable[[int], bytes], rsp_id: int) -> bytes:
	rsp_header = readfunc(4)
	msg_len, = struct.unpack_from("<H", rsp_header, 1)
	rsp_data = readfunc(msg_len - 4)
	validate_msg(rsp_header, rsp_data, rsp_id)
	return rsp_data

def make_cmd(cmd_len: int, id: int) -> bytearray:
	data = bytearray(cmd_len)
	struct.pack_into("<BHB", data, 0, HDR_START_CHAR, cmd_len, id)
	return data

def insert_crc(data: bytearray) -> None:
	crc = crc_4(data[0:len(data)-4])
	struct.pack_into("<I", data, len(data)-4, crc)