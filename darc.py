# Darc - A simple one-file database for large collections of tensor data.
#        Supports fast reading of crops of big tensors, accessing tensors by index or key, and most importantly,
#        supports reading the same database simultaneously from multiple processes.
#
# Change log:
#        Version 2.1: Add support for negative numbers in slices.
#                     Improve error reporting.
#
#        Version 2:   Add support for indexing by strings, and
#                     corresponding .keys() and .items().
#
#        Version 1:   Support multiple dtypes and variable dimensions.
#                     Support reading also when opened for writing.
#                     Fix an indexing issue for larger files.
#                     Support '-1' for chunk dimensions for maximum size.
#                   
#        Version 0:   Initial release.

import pdb
import struct
import os

import itertools
import numpy as np


SIGNATURE = b'darc'
VERSION = 2
DIRECTORY_ADDRESS_OFFSET = 8
NAME_TO_INDEX_ADDRESS_OFFSET = 16


# Defines dtype <-> int mapping.
_dtype_order = [np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64, np.complex64, np.complex128]

# Properties of dtypes.
_dtype_traits = {
	np.bool_: {'size': 1},	
	np.int8: {'size': 1},
	np.int16: {'size': 2},
	np.int32: {'size': 4},
	np.int64: {'size': 8},
	np.uint8: {'size': 1},
	np.uint16: {'size': 2},
	np.uint32: {'size': 4},
	np.uint64: {'size': 8},
	np.float16: {'size': 2},
	np.float32: {'size': 4},
	np.float64: {'size': 8},
	np.complex64: {'size': 8},
	np.complex128: {'size': 16}
}

# Mapping from dtype to int.
_dtype_to_int = dict(((dtype, index) for index, dtype in enumerate(_dtype_order)))

# Mapping from int to dtype.
_int_to_dtype = dict(((index, dtype) for index, dtype in enumerate(_dtype_order)))


class DarcException(Exception):
	pass


class Metadata:
	'''Data required for reading the data of an object from the database file.
	   Shares the file object with the DataArchive.'''
	def __init__(self, file, data_offset, shape, chunk_shape, dtype):
		self.file = file
		self.data_offset = data_offset
		self.shape = np.array(shape)
		self.chunk_shape = np.array(chunk_shape)
		self.dtype = dtype
		
	def __getitem__(self, slices):
		'''Numpy-style getter for multidimensional slices. Does not support the step parameter.'''
		
		target_shape = []
		window_begin = []
		window_end = []
		
		if isinstance(slices, int):
			slices = [slices]
		
		for dimension, slice_ in enumerate(slices):
			if type(slice_) == slice:
				# Slice.
				start = slice_.start if slice_.start is not None else 0
				stop = slice_.stop if slice_.stop is not None else self.shape[dimension]
				
				if start < 0:
					start += self.shape[dimension]
				if stop < 0:
					stop += self.shape[dimension]
				
				window_begin.append(start)
				window_end.append(stop)
				target_shape.append(stop - start)
				
				if slice_.step:
					raise DarcException("Steps not supported.")
			else:
				# Integer.
				index = int(slice_)
				if index < 0:
					index += self.shape[dimension]
					
				window_begin.append(index)
				window_end.append(index + 1)
					
		#crop_start = (
		#	slices[0].start if slices[0].start is not None else 0,
		#	slices[1].start if slices[1].start is not None else 0,
		#	slices[2].start if slices[2].start is not None else 0
		#)
		#
		#crop_stop = (
		#	slices[0].stop if slices[0].stop is not None else self.shape[0],
		#	slices[1].stop if slices[1].stop is not None else self.shape[1],
		#	slices[2].stop if slices[2].stop is not None else self.shape[2]
		#)
		
		result = self.data(crop=(window_begin, window_end))
		#pdb.set_trace()
		return result.reshape(target_shape)
	
	def data(self, crop=None):
		'''Reads all or a crop of the data from the database.'''
		if crop:
			if len(crop) != 2:
				raise DarcException('Invalid data window: should be tuple (begin, end).')
			if len(crop[0]) != len(self.shape) or len(crop[1]) != len(self.shape):
				raise DarcException('Dimensionality of requested crop [{}]-[{}] is incompatible with shape [{}].'.format(
					", ".join((repr(x) for x in crop[0])),
					", ".join((repr(x) for x in crop[1])),
					", ".join((repr(x) for x in self.shape))
				))
			
		self.file.seek(self.data_offset, 0)
		
		shape = self.shape
		chunk_shape = self.chunk_shape
		dimensions = len(shape)
		
		# Evaluate chunk structure of data.
		#chunk_counts = (
		#	(shape[0] + chunk_shape[0] - 1) // chunk_shape[0],
		#	(shape[1] + chunk_shape[1] - 1) // chunk_shape[1],
		#	(shape[2] + chunk_shape[2] - 1) // chunk_shape[2]
		#)
		chunk_counts = (shape + chunk_shape  - 1) // chunk_shape
		
		
		dtype_size = _dtype_traits[self.dtype]['size']
		#chunk_byte_size = chunk_shape[0] * chunk_shape[1] * chunk_shape[2] * dtype_size
		chunk_byte_size = np.prod(chunk_shape) * dtype_size
		
		#chunk_strides = (
		#	chunk_counts[2] * chunk_counts[1] * chunk_byte_size,
		#	chunk_counts[2] * chunk_byte_size,
		#	chunk_byte_size
		#)
		chunk_strides = chunk_byte_size * np.array([np.prod(chunk_counts[i:]) for i in range(1, dimensions+1)])

		# Calculate which chunks need to be read.
		if crop is None:
			# Read the whole data.
			crop_begin = np.zeros(shape=(dimensions,), dtype=np.int_)
			crop_end = shape
		else:
			# Read a crop.
			crop_begin = np.array(crop[0], dtype=np.int_)
			crop_end = np.array(crop[1], dtype=np.int_)
		
		#chunk_index_begin = (
		#	crop_begin[0] // chunk_shape[0],
		#	crop_begin[1] // chunk_shape[1],
		#	crop_begin[2] // chunk_shape[2]
		#)
		#chunk_index_end = (
		#	(crop_end[0] + chunk_shape[0] - 1) // chunk_shape[0],
		#	(crop_end[1] + chunk_shape[1] - 1) // chunk_shape[1],
		#	(crop_end[2] + chunk_shape[2] - 1) // chunk_shape[2]
		#)
		chunk_index_begin = crop_begin // chunk_shape
		chunk_index_end = (crop_end + chunk_shape - 1) // chunk_shape
		
		# Create the result array.
		result_shape = crop_end - crop_begin
		result = np.empty(shape=result_shape, dtype=self.dtype)
		
		# Iterate over chunks.
		chunk_begin = np.zeros(shape=(dimensions,), dtype=np.int_)
		chunk_end = np.zeros(shape=(dimensions,), dtype=np.int_)
		
		#on_edge = np.zeros(shape=(dimensions,), dtype=np.bool_)
		

		#for i in range(chunk_index_begin[0], chunk_index_end[0]):
		#	for j in range(chunk_index_begin[1], chunk_index_end[1]):
		#		for k in range(chunk_index_begin[2], chunk_index_end[2]):
		chunk_indices = (range(chunk_index_begin[d], chunk_index_end[d]) for d in range(dimensions))
		
		for it in itertools.product(*chunk_indices):
			index = np.array(it, dtype=np.int)
			
			# Read chunk from file.
			self.file.seek(self.data_offset + np.dot(index.astype(np.uint64), chunk_strides.astype(np.uint64)).item())
			
			chunk_bytes = self.file.read(chunk_byte_size)
			chunk = np.frombuffer(chunk_bytes, dtype=self.dtype)
			chunk = chunk.reshape(chunk_shape)
			
			# Evaluate the intersection of the chunk and the crop, in absolute coordinates.
			chunk_begin = index * chunk_shape
			#chunk_begin = *
			#	i * chunk_shape[0],
			#	j * chunk_shape[1],
			#	k * chunk_shape[2]
			#
			#)
			chunk_end = chunk_begin + chunk_shape
			#chunk_end = (
			#	chunk_begin[0] + chunk_shape[0],
			#	chunk_begin[1] + chunk_shape[1],
			#	chunk_begin[2] + chunk_shape[2],
			#)
			
			#if np.any(on_edge):
				# Probably only part of the chunk is needed.
				#intersection_begin = (
				#	max(chunk_begin[0], crop_begin[0]),
				#	max(chunk_begin[1], crop_begin[1]),
				#	max(chunk_begin[2], crop_begin[2])
				#)
			intersection_begin = np.maximum(chunk_begin, crop_begin)
				#intersection_end = (
				#	min(chunk_end[0], crop_end[0]),
				#	min(chunk_end[1], crop_end[1]),
				#	min(chunk_end[2], crop_end[2])
				#)
			intersection_end = np.minimum(chunk_end, crop_end)
				
			# Write the intersection into the result.
			result_window_begin = intersection_begin - crop_begin
			result_window_end = intersection_end - crop_begin
			result_slice = tuple(slice(result_window_begin[d], result_window_end[d]) for d in range(dimensions))
			
				
			if np.any(intersection_begin != chunk_begin) or any(intersection_end != chunk_end):
				chunk_window_begin = intersection_begin - chunk_begin
				chunk_window_end = intersection_end - chunk_begin
				chunk_slice = tuple(slice(chunk_window_begin[d], chunk_window_end[d]) for d in range(dimensions))
				
				result[result_slice] = chunk[chunk_slice]
			else:
				result[result_slice] = chunk
				
				#result[
				#	intersection_begin[0] - crop_begin[0] : intersection_end[0] - crop_begin[0],
				#	intersection_begin[1] - crop_begin[1] : intersection_end[1] - crop_begin[1],
				#	intersection_begin[2] - crop_begin[2] : intersection_end[2] - crop_begin[2]
				#] = chunk[
				#	intersection_begin[0] - chunk_begin[0] : intersection_end[0] - chunk_begin[0],
				#	intersection_begin[1] - chunk_begin[1] : intersection_end[1] - chunk_begin[1],
				#	intersection_begin[2] - chunk_begin[2] : intersection_end[2] - chunk_begin[2]
				#]
			#else:
			#	# Use the whole chunk.
			#	result[
			#		chunk_begin[0] - crop_begin[0] : chunk_end[0] - crop_begin[0],
			#		chunk_begin[1] - crop_begin[1] : chunk_end[1] - crop_begin[1],
			#		chunk_begin[2] - crop_begin[2] : chunk_end[2] - crop_begin[2]
			#	] = chunk
				
		
		# Finished!
		return result
		

class DataArchive:
	def __init__(self, path, mode='r'):
		self.opened = False
		self.version = VERSION
		
		self.directory = []
		self.name_to_index = {}
		
		self.mode = mode
		
		if mode == 'r':
			self.file = open(path, 'rb')
			self._readMetaData()
		elif mode == 'w':
			self.file = open(path, 'wb+')
			self._initWrite()
		else:
			raise DarcException("Unsupported mode.")
		
		self.opened = True
	
	def __del__(self):
		self.close()
	
	def __enter__(self):
		return self
	
	def __exit__(self, exc_type, exc_value, traceback):
		self.close()
		
	def _initWrite(self):
		self.file.write(SIGNATURE)
		self.file.write(struct.pack('<I', VERSION))
		self.file.write(struct.pack('<Q', 0))
		self.file.write(struct.pack('<Q', 0))
	
	def _readMetaData(self):
		# Read signature.
		signature = self.file.read(4)
		if signature != SIGNATURE:
			raise DarcException("Not a DataArchive.")
		
		# Read version.
		version, = struct.unpack('<I', self.file.read(4))
		if version > VERSION:
			raise DarcException("Unsupported version.")
		self.version = version
	
	
		# Read directory address.
		directory_address, = struct.unpack('<Q', self.file.read(8))
		
		# Read name-to-index address.
		if version >= 2:
			name_to_index_address, = struct.unpack('<Q', self.file.read(8))
		
		
		# Read directory.
		self.file.seek(directory_address, 0)
		
		directory_size, = struct.unpack('<Q', self.file.read(8))
		
		offsets = struct.unpack('<{}Q'.format(directory_size), self.file.read(8 * directory_size))
		self.directory = list(offsets)

		# Read name-to-index mapping.
		if version >= 2:
			self.file.seek(name_to_index_address, 0)
			self.name_to_index = np.load(self.file).item()
		
	def _writeDirectory(self):
		# Write the directory to the end of the file.
		self.file.seek(0, 2) # Seek to end.
		directory_address = self.file.tell()
		
		# Write directory size.
		self.file.write(struct.pack('<Q', len(self.directory)))
		
		# Write offsets of elements.
		for offset in self.directory:
			self.file.write(struct.pack('<Q', offset))
			
		# Store the address of the directory.
		self.file.seek(DIRECTORY_ADDRESS_OFFSET, 0)
		self.file.write(struct.pack('<Q', directory_address))
	
	def _writeNameToIndex(self):
		# Write the name-to-index mapping to the end of the file.
		self.file.seek(0, 2) # Seek to end.
		name_to_index_address = self.file.tell()
		
		# Write directory size.
		np.save(self.file, np.asarray(self.name_to_index))
		
		# Store the address of the directory.
		self.file.seek(NAME_TO_INDEX_ADDRESS_OFFSET, 0)
		self.file.write(struct.pack('<Q', name_to_index_address))	
		
	def close(self):
		if not self.opened:
			return
		
		self.opened = False
		
		if self.mode == 'w':
			self._writeDirectory()
			self._writeNameToIndex()
		
		self.file.close()
	
	
	def append(self, data, chunks=None, name=None):
		'''Adds a data element to the archive, divided in chunks of the given size.
		   If a dimension in 'chunks' is -1, maximum size will be used.
		   
		   If a 'name' string is given, makes the current object accessible by darc_object[name]
		   in addition to accessing by index.
	   '''
		shape = np.array(data.shape)
		
		if not chunks:
			chunk_shape = shape
		else:
			chunk_shape = np.array([chunks[i] if chunks[i] != -1 else shape[i] for i in range(len(shape))])
		
		dimensions = len(shape)
		
		# Calculate number of chunks required for the data.
		chunk_counts = (shape + chunk_shape - 1) // chunk_shape
		#chunk_counts = (
		#	(data.shape[0] + chunk_shape[0] - 1) // chunk_shape[0],
		#	(data.shape[1] + chunk_shape[1] - 1) // chunk_shape[1],
		#	(data.shape[2] + chunk_shape[2] - 1) // chunk_shape[2]
		#)
		
		# Store metadata.
		buffers = []
		
		buffers.append(struct.pack('<I', len(shape))) # Dimensions.
		buffers.append(struct.pack('<{}I'.format(dimensions), *shape)) # Height, width, depth.
		buffers.append(struct.pack('<{}I'.format(dimensions), *chunk_shape)) # Chunk height, width, depth.
		buffers.append(struct.pack('<I', _dtype_to_int[data.dtype.type])) # Data type.

		# Serialize chunks.
		chunk_indices = (range(chunk_counts[d]) for d in range(dimensions))
		
		for it in itertools.product(*chunk_indices):
			index = np.array(it, dtype=np.int)

		#for i in range(chunk_counts[0]):
		#	for j in range(chunk_counts[1]):
		#		for k in range(chunk_counts[2]):
		
			# Calculate chunk window.
			window_begin = index * chunk_shape
			#window_begin = (
			#	i * chunk_shape[0],
			#	j * chunk_shape[1],
			#	k * chunk_shape[2]
			#)
					
			window_end = window_begin + chunk_shape
			#window_end = (
				#window_begin[0] + chunk_shape[0],
				#window_begin[1] + chunk_shape[1],
				#window_begin[2] + chunk_shape[2],
			#)
					
			#if all(window_end[i] <= data.shape[i] for i in range(3)):
			if np.all(window_end <= shape):
				# Write full chunk.
				#view = data[
				#	window_begin[0] : window_end[0],
				#	window_begin[1] : window_end[1],
				#	window_begin[2] : window_end[2]
				#]				
				window_slice = tuple(slice(window_begin[d], window_end[d]) for d in range(dimensions))
				view = data[window_slice]
								
				buffers.append(view.tobytes())
			else:
				# Write chunk with zero fill.			
				#window_end = (
				#	min(window_end[0], data.shape[0]),
				#	min(window_end[1], data.shape[1]),
				#	min(window_end[2], data.shape[2]),
				#)
				window_end = np.minimum(window_end, shape)

				window_slice = tuple(slice(window_begin[d], window_end[d]) for d in range(dimensions))
				temp_data = np.zeros(shape=chunk_shape, dtype=data.dtype)

				temp_data_slice = tuple(slice(0, window_end[d] - window_begin[d]) for d in range(dimensions))
				temp_data[temp_data_slice] = data[window_slice]
				#temp_data[
				#	0 : window_end[0] - window_begin[0],
				#	0 : window_end[1] - window_begin[1],
				#	0 : window_end[2] - window_begin[2]
				#] = data[
				#	window_begin[0] : window_end[0],
				#	window_begin[1] : window_end[1],
				#	window_begin[2] : window_end[2]
				#]
						
				buffers.append(temp_data.tobytes())
					
		self.file.seek(0, 2) # Seek to end.
		data_offset = self.file.tell() # Store offset of new data.
		self.file.write(b''.join(buffers)) # Write metadata and data.
		
		# Store into the directory.
		self.directory.append(data_offset) 
				
		if name is not None:
			self.name_to_index[name] = len(self.directory) - 1

	def __len__(self):
		'''Returns the number of data objects stored.'''
		return len(self.directory)
		
	def __getitem__(self, index):
		'''Convenience function for 'read'.'''
		return self.read(index)
				
	def read(self, index):
		'''Reads metadata of an object from the database.
		   Index is either a number or a known name of an element.
		   
		   Access the actual data by the .data() method of the returned Metadata object.'''
		
		if self.mode == 'w':
			self.file.flush()
		
		# Map name to index.
		if isinstance(index, str):
			index = self.name_to_index[index]
	
		# Seek to the object's data.
		data_offset = self.directory[index]
		self.file.seek(data_offset, 0) 
		
		# Read metadata.
		dimensions, = struct.unpack('<I', self.file.read(4))
		if dimensions < 1:
			raise DarcException('Unsupported data dimensionality.')
		
		shape = struct.unpack('<{}I'.format(dimensions), self.file.read(dimensions * 4))
		chunk_shape = struct.unpack('<{}I'.format(dimensions), self.file.read(dimensions * 4))
		
		if self.version >= 1:
			dtype_index, = struct.unpack('<I', self.file.read(4))
			dtype = _int_to_dtype[dtype_index]
		else:
			dtype = np.uint8
			
		array_offset = self.file.tell()
		
		return Metadata(
			self.file,
			data_offset=array_offset,
			shape=shape,
			chunk_shape=chunk_shape,
			dtype=dtype
		)
		
	def keys(self):
		'''Returns a generator that iterates over the stored object names.'''
		return self.name_to_index.keys()
	
	def items(self):
		'''Returns a generator that iterates over the stored (object name, Metadata) pairs.
		   
		   Does not iterate over objects which have been stored without a name.
		'''
		for key in self.keys():
			yield (key, self[key])

if __name__ == '__main__':
	# Tests.
	write_archive = DataArchive('remove_me.darc', 'w')
	
	data1 = np.zeros(shape=(4, 4, 4), dtype=np.uint8)
	for i in range(4):
		for j in range(4):
			for k in range(4):
				data1[i, j, k] = 16 * i + 4 * j + k

	data2 = np.zeros(shape=(3, 4, 5, 6), dtype=np.float16)
	for i in range(3):
		for j in range(4):
			for k in range(5):
				for l in range(6):
					data2[i, j, k] = 10.0 * i + 1.0 * j + 0.1 * k + 0.01 * l
				
	write_archive.append(data1)
	write_archive.append(data1, chunks=(3, 3, 3), name="333")
	write_archive.append(data1, chunks=(4, 2, 3), name="423")
	write_archive.append(data1, chunks=(25, 25, 5))
	
	write_archive.append(data2)
	write_archive.append(data2, chunks=(1, 1, 1, 1))
	write_archive.append(data2, chunks=(2, 3, 4, 1), name="2341")
	write_archive.append(data2, chunks=(1, 5, 5, 5))

	write_archive.close()
	
	read_archive = DataArchive('remove_me.darc', 'r')
	
	assert write_archive.directory == read_archive.directory
	assert np.array_equal(read_archive.read(0).data(), data1)
	assert np.array_equal(read_archive[0][:,:,:], data1)
	assert np.array_equal(read_archive[1].data(), data1)
	assert np.array_equal(read_archive[2].data(), data1)
	assert np.array_equal(read_archive[3].data(), data1)
	
	assert np.array_equal(read_archive[4][:,:,:,:], data2)
	assert np.array_equal(read_archive[5].data(), data2)
	assert np.array_equal(read_archive[6].data(), data2)
	assert np.array_equal(read_archive[7].data(), data2)
	
	assert np.array_equal(read_archive[0][0:2, 1:3, 2:4], data1[0:2, 1:3, 2:4])
	assert np.array_equal(read_archive[1][0:2, 1:3, 2:4], data1[0:2, 1:3, 2:4])
	assert np.array_equal(read_archive[2][0:2, 1:3, 2:4], data1[0:2, 1:3, 2:4])
	assert np.array_equal(read_archive[3][0:2, 1:3, 2:4], data1[0:2, 1:3, 2:4])
	
	assert np.array_equal(read_archive[4][1, 0:2, 1:3, 2:4], data2[1, 0:2, 1:3, 2:4])
	assert np.array_equal(read_archive[5][0:1, 0:2, 1:3, 2:4], data2[0:1, 0:2, 1:3, 2:4])
	assert np.array_equal(read_archive[6][1:2, 0:2, 1:3, 2:4], data2[1:2, 0:2, 1:3, 2:4])
	assert np.array_equal(read_archive[7][:, 0:2, 1:3, 2:4], data2[:, 0:2, 1:3, 2:4])

	assert np.array_equal(read_archive[7][0, 1, 2, 3], data2[0, 1, 2, 3])
	assert np.array_equal(read_archive[7][0, 1, :, 3], data2[0, 1, :, 3])

	assert np.array_equal(read_archive["333"].data(), data1)
	assert np.array_equal(read_archive["423"].data(), data1)
	assert np.array_equal(read_archive["2341"].data(), data2)
	
	assert sorted(list(read_archive.keys())) == ['2341', '333', '423']
	
	x = sorted(list(read_archive.items()))
	assert x[0][0] == '2341' and x[1][0] == '333' and x[2][0] == '423'
	assert np.all(x[0][1].data() == data2) and np.all(x[1][1].data() == data1) and np.all(x[2][1].data() == data1)

	read_archive.close()
	
	os.unlink('remove_me.darc')
