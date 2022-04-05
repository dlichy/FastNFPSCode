import numpy as np
import random

class RandomImageSelector:
	def __init__(self, num_to_load, num_lights_in_dataset):
		self.num_to_load = num_to_load
		self.num_lights_in_dataset = num_lights_in_dataset
		
	def __call__(self):
		return random.choices(range(0,self.num_lights_in_dataset),k=self.num_to_load)
	
class SelectedImageSelector:
	def __init__(self, indices_to_load=[]):
		self.indices_to_load = indices_to_load
		
	def __call__(self):
		return self.indices_to_load
		
		
class NormalizeByMeanDepthSynthetic():
	def __call__(self, sample):
		depth = sample['depth']
		mask = sample['mask']
		mean_depth = np.mean( depth.squeeze()[mask.squeeze()>0.1] ).astype(np.float32)
		
		sample['mean_distance'] = mean_depth
		scale = 1/mean_depth
		sample['depth'] *= scale
		sample['scale'] = scale
		
		
		return sample
