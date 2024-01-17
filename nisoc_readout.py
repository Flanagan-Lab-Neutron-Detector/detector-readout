"""This module implements the serial communications protocol used by NR0 ('HPT readout')."""

from typing import Any, Callable
from collections.abc import Sequence
from abc import ABC, abstractmethod
import crcmod
import struct

# CRC

CRC_POLY = 0x104C11DB7
CRC_SEED = 0x100000007
CRC_XOR_OUT = 0

# CRCs are compatible with Renesas Synergy CRC hardware used the dumb way
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

def _validate_msg(header: bytearray, data: bytearray, exp_id: int) -> bool:
	# Validate start char and total length
	start_char, length, id = struct.unpack_from("<BHB", header, 0)
	if start_char != 0x7E:
		raise MessageValidationError(f"Wrong start char {start_char:02X}h", header, data)
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

def _read_rsp(readfunc: Callable[[int], bytes], rsp_id: int) -> bytes:
	rsp_header = readfunc(4)
	msg_len, = struct.unpack_from("<H", rsp_header, 1)
	rsp_data = readfunc(msg_len - 4)
	_validate_msg(rsp_header, rsp_data, rsp_id)
	return rsp_data

def _make_cmd(cmd_len: int, id: int) -> bytearray:
	data = bytearray(cmd_len)
	struct.pack_into("<BHB", data, 0, HDR_START_CHAR, cmd_len, id)
	return data

def _insert_crc(data: bytearray) -> None:
	crc = crc_4(data[0:len(data)-4])
	struct.pack_into("<I", data, len(data)-4, crc)

class ReadoutBase(ABC):
	"""
	Base class for readout implementations.
	"""

	# The function used to read data from the readout
	readfunc: Callable[[int], bytes] = None
	# The function used to write data to the readout
	writefunc: Callable[[bytes], None] = None

	def __init__(self, readfunc: Callable[[int], bytes], writefunc: Callable[[bytes], None]):
		self.readfunc = readfunc
		self.writefunc = writefunc

	@abstractmethod
	def ping(self):
		"""Ping the readout"""
		pass

	@abstractmethod
	def vt_get_bit_count_kpage(self, base_address: int, read_mv: int) -> list:
		"""Get count of set bits in 1024 pages"""
		pass

	@abstractmethod
	def erase_chip(self) -> None:
		"""Erase the entire chip"""
		pass

	@abstractmethod
	def erase_sector(self, sector_address: int) -> None:
		"""Erase a sector"""
		pass

	@abstractmethod
	def program_sector(self, sector_address: int, prog_value: int) -> None:
		"""Program a sector"""
		pass

	@abstractmethod
	def program_chip(self, prog_value: int) -> None:
		"""Program the entire chip"""
		pass

	@abstractmethod
	def get_sector_bit_count(self, base_address: int, read_mv: int) -> int:
		"""Get count of set bits in one sector"""
		pass

	@abstractmethod
	def read_data(self, base_address: int, vt_mode: bool=False, read_mv: int=4000) -> bytearray:
		"""Read data from a sector"""
		pass

	@abstractmethod
	def write_data(self, base_address: int, words: list) -> None:
		"""Write data to a sector"""
		pass

	@abstractmethod
	def read_word(self, address: int, vt_mode: bool=False, read_mv: int=4000) -> int:
		"""Read single word"""
		pass

	@abstractmethod
	def write_cfg(self, address: int, data: int) -> None:
		"""Write configuration word"""
		pass

	@abstractmethod
	def read_cfg(self, address: int) -> int:
		"""Read configuration word"""
		pass

	@abstractmethod
	def cfg_flash_enter(self) -> None:
		"""Enter CFG Flash access mode"""
		pass

	@abstractmethod
	def cfg_flash_exit(self) -> None:
		"""Exit CFG Flash access mode"""
		pass

	@abstractmethod
	def cfg_flash_read(self, address: int, count: int) -> bytes:
		"""Read from CFG flash"""
		pass

	@abstractmethod
	def cfg_flash_write(self, address: int, count: int, data: bytes) -> None:
		"""Write to CFG flash"""
		pass

	@abstractmethod
	def cfg_flash_erase(self, address: int, erase_type: int) -> None:
		"""Erase CFG flash"""
		pass

	@abstractmethod
	def ana_get_cal_counts(self):
		"""Get analog calibration"""
		pass

	@abstractmethod
	def ana_set_cal_counts(self) -> None:
		"""Set analog calibration"""
		pass

	@abstractmethod
	def ana_set_active_counts(self, unit: int, unit_counts: int) -> None:
		"""Set the active counts for an analog unit"""
		pass

class ReadoutDummy(ReadoutBase):
	"""
	Dummy Readout implementation
	"""

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

	def read_data(self, base_address: int, count: int, vt_mode: bool=False, read_mv: int=4000) -> bytearray:
		data = bytearray(count)
		for i in range(count):
			data[i] = 0xFA if i % 2 else 0x50
		return data

	def write_data(self, base_address: int, words) -> None:
		pass

	def read_word(self, address: int, vt_mode: bool=False, read_mv: int=4000) -> int:
		return 0xAA

	def write_cfg(self, address: int, data: int) -> None:
		pass

	def read_cfg(self, address: int) -> int:
		return 0x1111

	def cfg_flash_enter(self) -> None:
		pass

	def cfg_flash_exit(self) -> None:
		pass

	def cfg_flash_read(self, address: int, count: int) -> bytes:
		return bytes([5] * count)

	def cfg_flash_write(self, address: int, count: int, data: bytes) -> None:
		pass

	def cfg_flash_erase(self, address: int, erase_type: int) -> None:
		pass

	def cfg_flash_dev_info(self) -> tuple[int, int, int, int, bytes, int, int, int]:
		return 0, 0, 0, 0, b'1234', 0, 0, 0

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

class ReadoutNR0(ReadoutBase):
	"""
	NR0 Readout implementation
	"""

	def ping(self) -> tuple[int, str, int]:
		cmd_len = 8
		#rsp_len = 32
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_ping'])
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		rsp_data = _read_rsp(self.readfunc, MSG_IDS['rsp_ping'])
		uptime, version_bin, is_busy = struct.unpack_from("<I16sI", rsp_data, 0)
		version = version_bin.decode('utf-8', 'ignore').rstrip('\x00')

		return uptime, version, is_busy

	def vt_get_bit_count_kpage(self, base_address: int, read_mv: int) -> list:
		cmd_len = 16
		#rsp_len = 1032
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_vt_get_bit_count_kpage'])
		struct.pack_into("<II", cmd_data, 4, base_address, read_mv)
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		rsp_data = _read_rsp(self.readfunc, MSG_IDS['rsp_vt_get_bit_count_kpage'])

		# TODO(aidan): Figure out what this is supposed to do
		return [rsp_data[4+i] for i in range(1024)]

	def erase_chip(self) -> None:
		cmd_len = 8
		#rsp_len = 8
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_erase_chip'])
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		_ = _read_rsp(self.readfunc, MSG_IDS['rsp_erase_chip'])

	def erase_sector(self, sector_address: int) -> None:
		cmd_len = 12
		#rsp_len = 8
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_erase_sector'])
		struct.pack_into("<I", cmd_data, 4, sector_address)
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		_ = _read_rsp(self.readfunc, MSG_IDS['rsp_erase_sector'])

	def program_sector(self, sector_address: int, prog_value: int) -> None:
		cmd_len = 16
		#rsp_len = 8
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_program_sector'])
		struct.pack_into("<IHH", cmd_data, 4, sector_address, prog_value, 0)
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		_ = _read_rsp(self.readfunc, MSG_IDS['rsp_program_sector'])

	def program_chip(self, prog_value: int) -> None:
		cmd_len = 12
		#rsp_len = 8
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_program_chip'])
		struct.pack_into("<HH", cmd_data, 4, prog_value, 0)
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		_ = _read_rsp(self.readfunc, MSG_IDS['rsp_program_chip'])

	def get_sector_bit_count(self, base_address: int, read_mv: int) -> int:
		cmd_len = 16
		#rsp_len = 12
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_get_sector_bit_count'])
		struct.pack_into("<II", cmd_data, 4, base_address, read_mv)
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		rsp_data = _read_rsp(self.readfunc, MSG_IDS['rsp_get_sector_bit_count'])
		bits_set, = struct.unpack_from("<I", rsp_data, 0)

		return bits_set

	def read_data(self, base_address: int, count: int, vt_mode: bool=False, read_mv: int=4000) -> bytearray:
		cmd_len = 20
		#rsp_len = 1032
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_read_data'])
		struct.pack_into("<III", cmd_data, 4, base_address, 1 if vt_mode else 0, read_mv)
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		rsp_data = _read_rsp(self.readfunc, MSG_IDS['rsp_read_data'])

		# data = []
		# for i in range(0, 1024, 2):
		#	 word, = struct.unpack_from("<H", rsp_data, 4 + 2*i)
		#	 data.append(word)

		return bytearray(rsp_data[0:-4])

	def write_data(self, base_address: int, words: list) -> None:
		cmd_len = 1040
		#rsp_len = 8
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_write_data'])

		for i in range(len(words)):
			struct.pack_into("<H", cmd_data, 4, words[i])
		struct.pack_into("<II", cmd_data, 516, base_address, len(words))
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		_ = _read_rsp(self.readfunc, MSG_IDS['rsp_write_data'])

	def read_word(self, address: int, vt_mode: bool=False, read_mv: int=4000) -> int:
		raise NotImplementedError

	def write_cfg(self, address: int, data: int) -> None:
		raise NotImplementedError

	def read_cfg(self, address: int) -> int:
		raise NotImplementedError

	def cfg_flash_enter(self) -> None:
		raise NotImplementedError

	def cfg_flash_exit(self) -> None:
		raise NotImplementedError

	def cfg_flash_read(self, address: int, count: int) -> bytes:
		raise NotImplementedError

	def cfg_flash_write(self, address: int, count: int, data: bytes) -> None:
		raise NotImplementedError

	def cfg_flash_erase(self, address: int, erase_type: int) -> None:
		raise NotImplementedError

	def cfg_flash_dev_info(self) -> tuple[int, int, int, int, bytes, int, int, int]:
		raise NotImplementedError

	def ana_get_cal_counts(self) -> tuple[int, int, int, int]:
		cmd_len = 8
		#rsp_len = 16
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_ana_get_cal_counts'])
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		rsp_data = _read_rsp(self.readfunc, MSG_IDS['rsp_ana_get_cal_counts'])
		ce_10v_cts, reset_10v_cts, wp_acc_10v_cts, spare_10v_cts = struct.unpack_from("<HHHH", rsp_data, 0)

		return ce_10v_cts, reset_10v_cts, wp_acc_10v_cts, spare_10v_cts

	def ana_set_cal_counts(self, ce_10v_cts: int, reset_10v_cts: int, wp_acc_10v_cts: int, spare_10v_cts: int) -> None:
		cmd_len = 16
		#rsp_len = 8
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_ana_set_cal_counts'])
		struct.pack_into("<HHHH", cmd_data, 4, ce_10v_cts, reset_10v_cts, wp_acc_10v_cts, spare_10v_cts)
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		_ = _read_rsp(self.readfunc, MSG_IDS['rsp_ana_set_cal_counts'])

	analog_unit_map = {
		"ce" : 0,
		"reset" : 1,
		"wp_acc" : 2,
		"spare" : 3
	}

	def ana_set_active_counts(self, unit: int, unit_counts: int) -> None:
		# Analog units
		# CE#	  0
		# RESET#   1
		# WP/ACC#  2
		# SPARE	3
		cmd_len = 16
		#rsp_len = 8
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_ana_set_active_counts'])
		struct.pack_into("<II", cmd_data, 4, unit, unit_counts)
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		_ = _read_rsp(self.readfunc, MSG_IDS['rsp_ana_set_active_counts'])

class ReadoutNR1(ReadoutBase):
	"""
	NR1 Readout implementation
	"""

	def ping(self) -> tuple[int, str, int]:
		cmd_len = 8
		#rsp_len = 32
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_ping'])
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		rsp_data = _read_rsp(self.readfunc, MSG_IDS['rsp_ping'])
		uptime, version_bin, is_busy = struct.unpack_from("<I16sI", rsp_data, 0)
		reset_flags, task, task_state = struct.unpack_from("<III", rsp_data, 24)
		version = version_bin.decode('utf-8', 'ignore').rstrip('\x00')

		return uptime, version, is_busy, reset_flags, task, task_state

	def vt_get_bit_count_kpage(self, base_address: int, read_mv: int) -> list:
		cmd_len = 16
		#rsp_len = 1032
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_vt_get_bit_count_kpage'])
		struct.pack_into("<II", cmd_data, 4, base_address, read_mv)
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		rsp_data = _read_rsp(self.readfunc, MSG_IDS['rsp_vt_get_bit_count_kpage'])

		# TODO(aidan): Figure out what this is supposed to do
		return [rsp_data[4+i] for i in range(1024)]

	def erase_chip(self) -> None:
		cmd_len = 8
		#rsp_len = 8
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_erase_chip'])
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		_ = _read_rsp(self.readfunc, MSG_IDS['rsp_erase_chip'])

	def erase_sector(self, sector_address: int) -> None:
		cmd_len = 12
		#rsp_len = 8
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_erase_sector'])
		struct.pack_into("<I", cmd_data, 4, sector_address)
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		_ = _read_rsp(self.readfunc, MSG_IDS['rsp_erase_sector'])

	def program_sector(self, sector_address: int, prog_value: int) -> None:
		cmd_len = 16
		#rsp_len = 8
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_program_sector'])
		struct.pack_into("<IHH", cmd_data, 4, sector_address, prog_value, 0)
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		_ = _read_rsp(self.readfunc, MSG_IDS['rsp_program_sector'])

	def program_chip(self, prog_value: int) -> None:
		cmd_len = 12
		#rsp_len = 8
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_program_chip'])
		struct.pack_into("<HH", cmd_data, 4, prog_value, 0)
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		_ = _read_rsp(self.readfunc, MSG_IDS['rsp_program_chip'])

	def get_sector_bit_count(self, base_address: int, read_mv: int) -> int:
		cmd_len = 16
		#rsp_len = 12
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_get_sector_bit_count'])
		struct.pack_into("<II", cmd_data, 4, base_address, read_mv)
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		rsp_data = _read_rsp(self.readfunc, MSG_IDS['rsp_get_sector_bit_count'])
		bits_set, = struct.unpack_from("<I", rsp_data, 0)

		return bits_set

	def read_data(self, base_address: int, count: int, vt_mode: bool=False, read_mv: int=4000) -> bytearray:
		cmd_len = 24
		#rsp_len = 1032
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_read_data'])
		struct.pack_into("<IIII", cmd_data, 4, base_address, 1 if vt_mode else 0, read_mv, count)
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		rsp_data = _read_rsp(self.readfunc, MSG_IDS['rsp_read_data'])

		return bytearray(rsp_data[0:-4])

	def write_data(self, base_address: int, words: list) -> None:
		cmd_len = 1040
		#rsp_len = 8
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_write_data'])

		for i in range(len(words)):
			struct.pack_into("<H", cmd_data, 4, words[i])
		struct.pack_into("<II", cmd_data, 516, base_address, len(words))
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		_ = _read_rsp(self.readfunc, MSG_IDS['rsp_write_data'])

	def read_word(self, address: int, vt_mode: bool=False, read_mv: int=4000) -> int:
		cmd_len = 20
		#rsp_len = 12
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_read_word'])
		struct.pack_into("<III", cmd_data, 4, 1 if vt_mode else 0, read_mv, address)
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		rsp_data = _read_rsp(self.readfunc, MSG_IDS['rsp_read_word'])
		word, = struct.unpack_from("<H", rsp_data, 0)

		return word

	def write_cfg(self, address: int, data: int) -> None:
		cmd_len = 16
		#rsp_len = 8
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_write_cfg'])
		struct.pack_into("<II", cmd_data, 4, address, data)
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		_ = _read_rsp(self.readfunc, MSG_IDS['rsp_write_cfg'])

	def read_cfg(self, address: int) -> int:
		cmd_len = 12
		#rsp_len = 12
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_read_cfg'])
		struct.pack_into("<I", cmd_data, 4, address)
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		rsp_data = _read_rsp(self.readfunc, MSG_IDS['rsp_read_cfg'])
		word, = struct.unpack_from("<I", rsp_data, 0)

		return word

	def cfg_flash_enter(self) -> None:
		cmd_len = 8
		#rsp_len = 8
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_cfg_flash_enter'])
		# no data
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		_ = _read_rsp(self.readfunc, MSG_IDS['rsp_cfg_flash_enter'])

	def cfg_flash_exit(self) -> None:
		cmd_len = 8
		#rsp_len = 8
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_cfg_flash_exit'])
		# no data
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		_ = _read_rsp(self.readfunc, MSG_IDS['rsp_cfg_flash_exit'])

	def cfg_flash_read(self, address: int, count: int) -> bytes:
		cmd_len = 16
		#rsp_len = 8 + count
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_cfg_flash_read'])
		struct.pack_into("<II", cmd_data, 4, address, count)
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		rsp_data = _read_rsp(self.readfunc, MSG_IDS['rsp_cfg_flash_read'])

		return bytes(rsp_data[0:-4])

	def cfg_flash_write(self, address: int, count: int, data: bytes) -> None:
		cmd_len = 8 + 8 + count
		#rsp_len = 8
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_cfg_flash_write'])
		struct.pack_into(f"<II{count}s", cmd_data, 4, address, count, data)
		_insert_crc(cmd_data)
		#print(f"SEND cmd_cfg_flash_write {MSG_IDS['cmd_cfg_flash_write']} {address=} {count=} {data=} {cmd_data=}")

		self.writefunc(cmd_data)
		_ = _read_rsp(self.readfunc, MSG_IDS['rsp_cfg_flash_write'])

	def cfg_flash_erase(self, address: int, erase_type: int) -> None:
		cmd_len = 16
		#rsp_len = 8
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_cfg_flash_erase'])
		# erase_type 0=4k 1=32k 2=64k 3=chip
		struct.pack_into("<II", cmd_data, 4, address, erase_type)
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		_ = _read_rsp(self.readfunc, MSG_IDS['rsp_cfg_flash_erase'])

	def cfg_flash_dev_info(self) -> tuple[int, int, int, int, bytes, int, int, int]:
		cmd_len = 8
		#rsp_len = 16
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_cfg_flash_dev_info'])
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		rsp_data = _read_rsp(self.readfunc, MSG_IDS['rsp_cfg_flash_dev_info'])
		mfg, devid, jtype, jcap, uid, sr1, sr2, sr3 = struct.unpack_from("<BBBB4sBBB", rsp_data, 0)

		return mfg, devid, jtype, jcap, uid, sr1, sr2, sr3

	def ana_get_cal_counts(self, unit: int) -> tuple[int, int, int, int]:
		cmd_len = 12
		#rsp_len = 16
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_ana_get_cal_counts'])
		struct.pack_into("<I", cmd_data, 4, unit)
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		rsp_data = _read_rsp(self.readfunc, MSG_IDS['rsp_ana_get_cal_counts'])
		calc0, calc1 = struct.unpack_from("<ff", rsp_data, 0)

		return calc0, calc1

	def ana_set_cal_counts(self, unit: int, calc0: float, calc1: float) -> None:
		cmd_len = 20
		#rsp_len = 8
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_ana_set_cal_counts'])
		struct.pack_into("<Iff", cmd_data, 4, unit, calc0, calc1)
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		_ = _read_rsp(self.readfunc, MSG_IDS['rsp_ana_set_cal_counts'])

	analog_unit_map = {
		"ce" : 0,
		"reset" : 1,
		"wp_acc" : 2,
		"spare" : 3
	}

	def ana_set_active_counts(self, unit: int, unit_counts: int) -> None:
		# Analog units
		# CE#	  0
		# RESET#   1
		# WP/ACC#  2
		# SPARE	3
		cmd_len = 16
		#rsp_len = 8
		cmd_data = _make_cmd(cmd_len, MSG_IDS['cmd_ana_set_active_counts'])
		struct.pack_into("<II", cmd_data, 4, unit, unit_counts)
		_insert_crc(cmd_data)

		self.writefunc(cmd_data)
		_ = _read_rsp(self.readfunc, MSG_IDS['rsp_ana_set_active_counts'])
