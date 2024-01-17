"""Readout driver abstract base class"""

from typing import Callable
from abc import ABC, abstractmethod

class ReadoutBase(ABC):
	"""
	Base class for readout implementations.
	"""

	# The function used to read data from the readout
	readfunc: Callable[[int], bytes]# = None
	# The function used to write data to the readout
	writefunc: Callable[[bytes], None]# = None

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