"""Readout driver for NR0"""

import struct
from .base import ReadoutBase
from .messaging import MSG_IDS, make_cmd, insert_crc, read_rsp

class ReadoutNR0(ReadoutBase):
	"""
	NR0 Readout implementation
	"""

	def ping(self) -> tuple[int, str, int]:
		cmd_len = 8
		#rsp_len = 32
		cmd_data = make_cmd(cmd_len, MSG_IDS['cmd_ping'])
		insert_crc(cmd_data)

		self.writefunc(cmd_data)
		rsp_data = read_rsp(self.readfunc, MSG_IDS['rsp_ping'])
		uptime, version_bin, is_busy = struct.unpack_from("<I16sI", rsp_data, 0)
		version = version_bin.decode('utf-8', 'ignore').rstrip('\x00')

		return uptime, version, is_busy

	def vt_get_bit_count_kpage(self, base_address: int, read_mv: int) -> list:
		cmd_len = 16
		#rsp_len = 1032
		cmd_data = make_cmd(cmd_len, MSG_IDS['cmd_vt_get_bit_count_kpage'])
		struct.pack_into("<II", cmd_data, 4, base_address, read_mv)
		insert_crc(cmd_data)

		self.writefunc(cmd_data)
		rsp_data = read_rsp(self.readfunc, MSG_IDS['rsp_vt_get_bit_count_kpage'])

		# TODO(aidan): Figure out what this is supposed to do
		return [rsp_data[4+i] for i in range(1024)]

	def erase_chip(self) -> None:
		cmd_len = 8
		#rsp_len = 8
		cmd_data = make_cmd(cmd_len, MSG_IDS['cmd_erase_chip'])
		insert_crc(cmd_data)

		self.writefunc(cmd_data)
		_ = read_rsp(self.readfunc, MSG_IDS['rsp_erase_chip'])

	def erase_sector(self, sector_address: int) -> None:
		cmd_len = 12
		#rsp_len = 8
		cmd_data = make_cmd(cmd_len, MSG_IDS['cmd_erase_sector'])
		struct.pack_into("<I", cmd_data, 4, sector_address)
		insert_crc(cmd_data)

		self.writefunc(cmd_data)
		_ = read_rsp(self.readfunc, MSG_IDS['rsp_erase_sector'])

	def program_sector(self, sector_address: int, prog_value: int) -> None:
		cmd_len = 16
		#rsp_len = 8
		cmd_data = make_cmd(cmd_len, MSG_IDS['cmd_program_sector'])
		struct.pack_into("<IHH", cmd_data, 4, sector_address, prog_value, 0)
		insert_crc(cmd_data)

		self.writefunc(cmd_data)
		_ = read_rsp(self.readfunc, MSG_IDS['rsp_program_sector'])

	def program_chip(self, prog_value: int) -> None:
		cmd_len = 12
		#rsp_len = 8
		cmd_data = make_cmd(cmd_len, MSG_IDS['cmd_program_chip'])
		struct.pack_into("<HH", cmd_data, 4, prog_value, 0)
		insert_crc(cmd_data)

		self.writefunc(cmd_data)
		_ = read_rsp(self.readfunc, MSG_IDS['rsp_program_chip'])

	def get_sector_bit_count(self, base_address: int, read_mv: int) -> int:
		cmd_len = 16
		#rsp_len = 12
		cmd_data = make_cmd(cmd_len, MSG_IDS['cmd_get_sector_bit_count'])
		struct.pack_into("<II", cmd_data, 4, base_address, read_mv)
		insert_crc(cmd_data)

		self.writefunc(cmd_data)
		rsp_data = read_rsp(self.readfunc, MSG_IDS['rsp_get_sector_bit_count'])
		bits_set, = struct.unpack_from("<I", rsp_data, 0)

		return bits_set

	def read_data(self, base_address: int, count: int, vt_mode: bool=False, read_mv: int=4000) -> bytearray:
		cmd_len = 20
		#rsp_len = 1032
		cmd_data = make_cmd(cmd_len, MSG_IDS['cmd_read_data'])
		struct.pack_into("<III", cmd_data, 4, base_address, 1 if vt_mode else 0, read_mv)
		insert_crc(cmd_data)

		self.writefunc(cmd_data)
		rsp_data = read_rsp(self.readfunc, MSG_IDS['rsp_read_data'])

		# data = []
		# for i in range(0, 1024, 2):
		#	 word, = struct.unpack_from("<H", rsp_data, 4 + 2*i)
		#	 data.append(word)

		return bytearray(rsp_data[0:-4])

	def write_data(self, base_address: int, words: list) -> None:
		cmd_len = 1040
		#rsp_len = 8
		cmd_data = make_cmd(cmd_len, MSG_IDS['cmd_write_data'])

		for i in range(len(words)):
			struct.pack_into("<H", cmd_data, 4, words[i])
		struct.pack_into("<II", cmd_data, 516, base_address, len(words))
		insert_crc(cmd_data)

		self.writefunc(cmd_data)
		_ = read_rsp(self.readfunc, MSG_IDS['rsp_write_data'])

	def read_word(self, address: int, samples: int, vt_mode: bool=False, read_mv: int=4000) -> tuple[int, tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]]:
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
		cmd_data = make_cmd(cmd_len, MSG_IDS['cmd_ana_get_cal_counts'])
		insert_crc(cmd_data)

		self.writefunc(cmd_data)
		rsp_data = read_rsp(self.readfunc, MSG_IDS['rsp_ana_get_cal_counts'])
		ce_10v_cts, reset_10v_cts, wp_acc_10v_cts, spare_10v_cts = struct.unpack_from("<HHHH", rsp_data, 0)

		return ce_10v_cts, reset_10v_cts, wp_acc_10v_cts, spare_10v_cts

	def ana_set_cal_counts(self, ce_10v_cts: int, reset_10v_cts: int, wp_acc_10v_cts: int, spare_10v_cts: int) -> None:
		cmd_len = 16
		#rsp_len = 8
		cmd_data = make_cmd(cmd_len, MSG_IDS['cmd_ana_set_cal_counts'])
		struct.pack_into("<HHHH", cmd_data, 4, ce_10v_cts, reset_10v_cts, wp_acc_10v_cts, spare_10v_cts)
		insert_crc(cmd_data)

		self.writefunc(cmd_data)
		_ = read_rsp(self.readfunc, MSG_IDS['rsp_ana_set_cal_counts'])

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
		cmd_data = make_cmd(cmd_len, MSG_IDS['cmd_ana_set_active_counts'])
		struct.pack_into("<II", cmd_data, 4, unit, unit_counts)
		insert_crc(cmd_data)

		self.writefunc(cmd_data)
		_ = read_rsp(self.readfunc, MSG_IDS['rsp_ana_set_active_counts'])