import numpy as np
import torch
from data_processing.crop_pad_intrinsics import *


class CropBasedOnMask:
	def __init__(self,mask_key, boarder, square=False, keys=[], list_keys=[]):
		self.mask_key = mask_key
		self.boarder=boarder
		self.keys = keys
		self.list_keys=list_keys
		self.square=square
		
	def __call__(self, sample):
		mask = sample[self.mask_key]
		
		indices = np.argwhere(mask.squeeze())


		left = indices[:,1].min()
		right = indices[:,1].max()+1
		top = indices[:,0].min()
		bottom = indices[:,0].max()+1
		
		if self.square:
			width = right-left
			height = bottom-top
			if height > width:
				pad = int((height-width)/2)
				left = left-pad
				right = right+pad
			else:
				pad = int((width-height)/2)
				top = top-pad
				bottom = bottom+pad
		
		intrinsics = sample['intrinsics']
		
		
		new_intrinsics  = crop_intrinsics_normalized((top,bottom),(left,right),intrinsics, mask.shape)
		sample['intrinsics'] = new_intrinsics
		
		for key in self.keys:
			cropped = crop_image((top,bottom),(left,right), sample[key], channel_last = True)
			sample[key] = cropped
		
		for list_key in self.list_keys:
			for i,image in enumerate(sample[list_key]):
				cropped = crop_image((top,bottom),(left,right), image, channel_last = True)
				sample[list_key][i] = cropped
		
		return sample
		
		
	
	
	
	
	
