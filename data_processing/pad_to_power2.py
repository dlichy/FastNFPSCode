import data_processing.optical_flow_funs as OF
from data_processing.crop_pad_intrinsics import *
import torch
import numpy as np


class PadSquareToPower2Intrinsics:
	def __init__(self,keys=[],list_keys=[]):
		self.keys = keys
		self.list_keys = list_keys
	
	def __call__(self,sample):
		if len(self.keys) > 0:
			s = sample[self.keys[0]].shape[0:2]
		else: 
			s = sample[self.list_keys[0]][0].shape[0:2]
	
		new_size = 2**np.ceil(np.log2(s))
		new_size = int(np.max(new_size))
		

		top_pad = int((new_size-s[0])/2)
		bottom_pad = int(new_size-s[0] - top_pad)
		left_pad = int((new_size-s[1])/2)
		right_pad = int(new_size-s[1] - left_pad)
		
		intrinsics = sample['intrinsics']
		new_intrinsics = pad_intrinsics_normalized((top_pad,bottom_pad), (left_pad,right_pad), intrinsics, s)
		sample['intrinsics'] = new_intrinsics
			
		

		for key in self.keys:
			image = sample[key]
			if image.ndim == 3:
				pad_shape = ((top_pad,bottom_pad),(left_pad,right_pad),(0,0))
			else:
				pad_shape = ((top_pad,bottom_pad),(left_pad,right_pad))
			sample[key] = np.pad(image,pad_shape)
			
		for list_key in self.list_keys:
			for i,image in enumerate(sample[list_key]):
				if image.ndim == 3:
					pad_shape =  ((top_pad,bottom_pad),(left_pad,right_pad),(0,0))
				else:
					pad_shape = ((top_pad,bottom_pad),(left_pad,right_pad))

				sample[list_key][i] = np.pad(image,pad_shape)
		
		
		return sample
