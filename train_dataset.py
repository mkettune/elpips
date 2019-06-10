import os
import imageio
import numpy as np
import skimage.transform
import concurrent.futures

import darc


def load_image(path):
	_, ext = os.path.splitext(path)
	if ext.lower() == '.npy':
		image = np.load(path)
	elif ext.lower() in ('.png', '.jpg'):
		image = imageio.imread(path).astype(np.float32) / 255.0
	else:
		raise Exception('Unknown image type.')
		
	return image

def load_image_uint(path):
	_, ext = os.path.splitext(path)
	if ext.lower() in ('.png', '.jpg'):
		image = imageio.imread(path)
	else:
		raise Exception('Unknown image type.')
		
	return image

def cached_listdir(directory):
	import pickle
	key = directory
	key = key.replace("/", "_")
	key = key.replace("\\", "_")
		
	pickle_file = "cached_listdir." + key + ".pickle"
	if os.path.exists(pickle_file):
		print("Reading cached dirlist for '{}'.".format(directory))
		files = pickle.load(open(pickle_file, 'rb'))
	else:
		print("Reading dirlist for '{}'.".format(directory))
		files = os.listdir(directory)
		pickle.dump(files, open(pickle_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

	return files


class Dataset:
	def __init__(self, directory, dataset_mode, load_size):
		self.directory = directory
		
		if not os.path.isdir('dataset'):
			raise Exception('Could not find the dataset directory \'dataset\'. Download the dataset first.')
			
		self.full_path = os.path.join('dataset', '2afc', directory)
		self.dataset_mode = dataset_mode
		self.load_size = load_size
		self.cache = {'judge': [], 'p0': [], 'p1': [], 'ref': [], 'judge_path': [], 'p0_path': [], 'p1_path': [], 'ref_path': []}

		darc_path = self.getDarcPath()		
		if not os.path.exists(darc_path):		
			self.createDarc()
		self.darc = darc.DataArchive(darc_path)
		
	def __len__(self):
		return len(self.darc) // 2
		
	def getCacheKey(self):
		key = self.directory
		key = key.replace("/", "_")
		key = key.replace("\\", "_")
		return key
	
	def getDarcPath(self):
		return "cached_dataset.{}.{}.darc".format(self.getCacheKey(), self.load_size)
		
	def createDarc(self):
		'''Loads a dataset of images from disk and stores it in the Darc format.
		   Darc is similar in idea to HDF5 but much simpler and should handle multi-process parallelization issues better.'''
		print("Reading dataset '{}' from disk.".format(self.getCacheKey()))
		
		archive = darc.DataArchive(self.getDarcPath(), 'w')
		
		judge_files = cached_listdir(os.path.join(self.full_path, 'judge'))
		judge_files = [os.path.join(self.full_path, 'judge', file) for file in judge_files if os.path.splitext(file)[1] == '.npy']
		judge_files = sorted(judge_files)
		
		p0_files = cached_listdir(os.path.join(self.full_path, 'p0'))
		p0_files = [os.path.join(self.full_path, 'p0', file) for file in p0_files if os.path.splitext(file)[1] == '.png']
		p0_files = sorted(p0_files)
		
		p1_files = cached_listdir(os.path.join(self.full_path, 'p1'))
		p1_files = [os.path.join(self.full_path, 'p1', file) for file in p1_files if os.path.splitext(file)[1] == '.png']
		p1_files = sorted(p1_files)
	
		ref_files = cached_listdir(os.path.join(self.full_path, 'ref'))
		ref_files = [os.path.join(self.full_path, 'ref', file) for file in ref_files if os.path.splitext(file)[1] == '.png']
		ref_files = sorted(ref_files)
		
		def handle_task(judge_path, p0_path, p1_path, ref_path):
			judge = np.load(judge_path)[0]
			
			# Modify to use this code to use exactly the same data as the original LPIPS paper (as far as we know).
			#
			# This applies a bilinear 4x downscaling to the input images of resolution 256x256, which in reality
			# are just 64x64 images upscaled to 256x256 with nearest neighbor. We do not know why this is the case.
			#
			# A couple of the images are, for some reason, 252x252, and there's a slight difference in the images in that case.
			#####
			#import PIL
			#
			#p0 = PIL.Image.open(p0_path).convert('RGB')
			#p0 = p0.resize((self.load_size, self.load_size), PIL.Image.BILINEAR)
			#p0 = np.array(p0) / 255.0
			#
			#p1 = PIL.Image.open(p1_path).convert('RGB')
			#p1 = p1.resize((self.load_size, self.load_size), PIL.Image.BILINEAR)
			#p1 = np.array(p1) / 255.0
			#
			#ref = PIL.Image.open(ref_path).convert('RGB')
			#ref = ref.resize((self.load_size, self.load_size), PIL.Image.BILINEAR)
			#ref = np.array(ref) / 255.0
			#
			#p0 = p0[:,:,0:3]
			#p1 = p1[:,:,0:3]
			#ref = ref[:,:,0:3]
			#####
			
			p0 = load_image_uint(p0_path)
			p1 = load_image_uint(p1_path)
			ref = load_image_uint(ref_path)
			
			assert len(p0.shape) >= 3 and p0.shape[2] >= 3
			assert len(p1.shape) >= 3 and p1.shape[2] >= 3
			assert len(ref.shape) >= 3 and ref.shape[2] >= 3
			
			p0 = p0[:,:,0:3]
			p1 = p1[:,:,0:3]
			ref = ref[:,:,0:3]
			
			if self.dataset_mode == '2afc' and p0.shape[0:2] != (self.load_size, self.load_size):
				p0 = skimage.transform.resize(p0, [self.load_size, self.load_size, 3], mode='reflect', anti_aliasing=False)
				p1 = skimage.transform.resize(p1, [self.load_size, self.load_size, 3], mode='reflect', anti_aliasing=False)
				ref = skimage.transform.resize(ref, [self.load_size, self.load_size, 3], mode='reflect', anti_aliasing=False)
            
				# Note: Images now in [0, 1].
				
			return judge, p0, p1, ref, judge_path, p0_path, p1_path, ref_path
		
		# Check that each dataset sample has judge, p0, p1 and reference.
		if not all(len(collection) == len(judge_files) for collection in (p0_files, p1_files, ref_files)):
			raise Exception('Dataset files missing!')
			
		with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
			postponed_image_sets = []
			for i in range(len(judge_files)):
				judge_path, p0_path, p1_path, ref_path = judge_files[i], p0_files[i], p1_files[i], ref_files[i]
				postponed_image_sets.append((judge_path, p0_path, p1_path, ref_path))

			queued_image_sets = []
			i = 0
			while queued_image_sets or postponed_image_sets:
				while len(queued_image_sets) < 20 and postponed_image_sets:
					judge_path, p0_path, p1_path, ref_path = postponed_image_sets[0]
					queued_image_sets.append(executor.submit(handle_task, judge_path, p0_path, p1_path, ref_path))
					postponed_image_sets.pop(0)
				
				judge, p0, p1, ref, judge_path, p0_path, p1_path, ref_path = queued_image_sets[0].result()
				queued_image_sets.pop(0)
				
				print("{}/{}".format(i, len(judge_files)))
				
				p_tensor = np.stack([p0, p1, ref])
				archive.append(p_tensor, chunks=[1,-1,-1,-1], name="{}_p".format(i))
				
				judge_tensor = np.asarray(judge).reshape([1])
				archive.append(judge_tensor, name="{}_judge".format(i))
				
				i += 1

		archive.close()
		
	def __getitem__(self, index):
		return (
			self.darc["{}_p".format(index)],
			self.darc["{}_judge".format(index)]
		)
	

class DataLoader:
	def __init__(self, datasets, dataset_mode='2afc', load_size=64, batch_size=20, serial_batches=True):
		'''Reads a dataset and provides a generator interface to its images.
		
		   Parameters:
		       datasets:        A list of Dataset objects to load as one.
			   dataset_mode:    Only mode '2afc' is supported.
			   load_size:       Resolution into which the images are resampled; 64 is the correct value.
			   batch_size:      Size of returned minibatches.
			   serial_batches:  If true, images are returned in index order; otherwise random order is used.
		'''
		self.datasets = []
		for directory in datasets:
			self.datasets.append(Dataset(directory, dataset_mode, load_size=load_size))
		
		self.dataset_mode = dataset_mode
		self.load_size = load_size
		self.batch_size = batch_size
		self.serial_batches = serial_batches
		self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=100)
		
	def __len__(self):
		return sum(len(dataset) for dataset in self.datasets)
	
	def _getDatasetByIndex(self, index):
		current_index = index
		for dataset in self.datasets:
			if current_index - len(dataset) < 0:
				return dataset, current_index
			else:
				current_index -= len(dataset)
				
		raise Exception('Index out of range')

	def epoch(self):
		'''Generator that iterates over the minibatches of an epoch.'''
		import concurrent.futures
	
		# Get a full batch of indices, randomized if so requested.
		batch = list(range(len(self)))
		
		if not self.serial_batches:
			import random
			random.shuffle(batch)
		
		# Function for parallel reading of minibatches.
		_darcs = {}
		
		import threading
		def handle_task(indices):
			'''Constructs a minibatch with the given indices.'''
			thread_id = threading.get_ident()
			if not thread_id in _darcs:
				_darcs[thread_id] = {}
			
			judges, p0s, p1s, refs = [], [], [], []
			for i in indices:
				dataset, index = self._getDatasetByIndex(i)
				dataset_path = dataset.getDarcPath()
				if not dataset_path in _darcs[thread_id]:
					# One Darc per thread.
					_darcs[thread_id][dataset_path] = darc.DataArchive(dataset_path)
				db = _darcs[thread_id][dataset_path]
				
				p_data = db["{}_p".format(index)]
				judge_data = db["{}_judge".format(index)]
				
				judges.append(judge_data.data())
				
				p0s.append(p_data[0, :, :, :])
				p1s.append(p_data[1, :, :, :])
				refs.append(p_data[2, :, :, :])
				
			return {
				'judge': np.concatenate(judges),
				'p0': np.stack(p0s),
				'p1': np.stack(p1s),
				'ref': np.stack(refs)
			}

		# Read minibatches in parallel and yield a single minibatch at a time.
		self.postponed_batches = []
		
		indices = []
		for i in batch:
			indices.append(i)
			if len(indices) == self.batch_size:
				self.postponed_batches.append(indices)
				indices = []
		
		self.queued_batches = []
		while self.queued_batches or self.postponed_batches:
			while len(self.queued_batches) < 100 and self.postponed_batches:
				self.queued_batches.append(self.executor.submit(handle_task, self.postponed_batches[0]))
				self.postponed_batches.pop(0)
			
			for i, task in enumerate(self.queued_batches):
				if task.done():
					result = task.result()
					self.queued_batches.pop(i)
					yield result
					break
			