import numpy as np
import torch
import torchvision
from torchvision import transforms
import scipy.ndimage
import random
import cv2
from data_processing.crop_pad_intrinsics import *
import data_processing.optical_flow_funs as OF




class ErodeMask:
	def __init__(self,mask_erosion_size=(6,6),keys=[], list_keys=[],mask_thres=0.01):
		
		self.mask_erosion_size = mask_erosion_size
		self.keys = keys
		self.list_keys= list_keys
		self.mask_thres = mask_thres
		
	def __call__(self,sample):
		
		
		for key in self.keys:
			mask = sample[key]
			mask_eroded = scipy.ndimage.binary_erosion( (mask.squeeze() > self.mask_thres),structure=np.ones(self.mask_erosion_size))
			sample[key] = np.expand_dims(mask_eroded.astype('single'),axis=2)
		
		for list_key in self.list_keys:
			for i,mask in enumerate(sample[list_key]):
				mask_eroded = scipy.ndimage.binary_erosion( (mask.squeeze() > self.mask_thres),structure=np.ones(self.mask_erosion_size))
				sample[list_key][i] = np.expand_dims(mask_eroded.astype('single'),axis=2)
			
		return sample
		


class FillHoles:
	def __init__(self,keys=[],mask_thres=0.01):
		
		self.keys = keys
		self.mask_thres = mask_thres
		
	def __call__(self,sample):
		
		
		for key in self.keys:
			mask = sample[key]
			mask_filled = scipy.ndimage.binary_fill_holes( (mask.squeeze() > self.mask_thres) )
			sample[key+'_filled'] = np.expand_dims(mask_filled.astype('single'),axis=2)
		
		return sample


		
class MyToTensor:
	def __init__(self,keys=[],list_keys=[]):
		self.toTensor = transforms.ToTensor()
		self.keys = keys
		self.list_keys = list_keys
		
	def __call__(self,sample):
		for key in self.keys:
			sample[key] = self.toTensor(sample[key])
		
		for list_key in self.list_keys:
			for i,image in enumerate(sample[list_key]):
				sample[list_key][i] = self.toTensor(image)
			
		return sample

	
#random crop resize and adjust intrinsics accordingly		
class RandomCropResize:
	def __init__(self, output_size, keys=[], list_keys=[], scale=(0.5,1), ratio=(1,1)):
		self.scale = scale
		self.ratio = ratio
		self.output_size = output_size
		self.keys = keys
		self.list_keys = list_keys
		
	def __call__(self,sample):
		if len(self.keys) > 0:
			s = sample[self.keys[0]].shape
		else: 
			s = sample[self.list_keys[0]][0].shape
		
		input_width = s[1]
		input_height = s[0]
		
		aspect_ratio = random.uniform(*self.ratio)
		width = int(input_width*random.uniform(*self.scale))
		height = int(aspect_ratio*width)
		
		top = random.randrange(0,input_height-height+1)
		left = random.randrange(0,input_width-width+1)
		
		
		intrinsics = sample['intrinsics']
		new_intrinsics  = crop_intrinsics_normalized((top,top+height),(left,left+width),intrinsics, s[0:2])

		sample['intrinsics'] = new_intrinsics
		
		for key in self.keys:
			cropped = crop_image((top,top+height),(left,left+width), sample[key], channel_last = True)
			sample[key] = cv2.resize(cropped.astype(np.float32),dsize=self.output_size,interpolation = cv2.INTER_LINEAR)
		
		for list_key in self.list_keys:
			for i,image in enumerate(sample[list_key]):
				cropped = crop_image((top,top+height),(left,left+width), image, channel_last = True)
					
				sample[list_key][i] = cv2.resize(cropped.astype(np.float32),dsize=self.output_size,interpolation = cv2.INTER_LINEAR)
		
		
		return sample


		
class RandomScale:
	def __init__(self,keys=[], list_keys=[],my_range=(0.6,1.4)):
		self.range = my_range
		self.keys = keys
		self.list_keys = list_keys
		
	def __call__(self,sample):
		scale = random.uniform(*self.range)
		for key in self.keys:
			sample[key] = scale*sample[key]
		
		for list_key in self.list_keys:
			for i,image in enumerate(sample[list_key]):
				sample[list_key][i] = scale*image
				 
		return sample
		
class RandomScaleUnique:
	def __init__(self,keys=[], list_keys=[],my_range=(0.3,1.8)):
		self.range = my_range
		self.keys = keys
		self.list_keys = list_keys
		
	def __call__(self,sample):
		for key in self.keys:
			scale = random.uniform(*self.range)
			sample[key] = scale*sample[key]
		
		for list_key in self.list_keys:
			for i,image in enumerate(sample[list_key]):
				scale = random.uniform(*self.range)
				sample[list_key][i] = scale*image
				 
		return sample
		
class RandomScaleUniqueGaussian:
	def __init__(self,keys=[], list_keys=[],mu=1,sigma=0):
		self.mu = mu
		self.sigma = sigma
		self.keys = keys
		self.list_keys = list_keys
		
	def __call__(self,sample):
		for key in self.keys:
			scale = random.normalvariate(self.mu,self.sigma)
			sample[key] = scale*sample[key]
		
		for list_key in self.list_keys:
			for i,image in enumerate(sample[list_key]):
				scale = random.normalvariate(self.mu,self.sigma)
				sample[list_key][i] = scale*image
				 
		return sample		
		
		
class NormalizeByMedian:
	def __init__(self,keys=[],list_keys=[]):
		self.keys = keys
		self.list_keys = list_keys
	
	def __call__(self,sample):
		for key in self.keys:
			image = sample[key]
			scale = 1/(np.median(image).clip(min=0.001))
			sample[key] = scale*image
		
		for list_key in self.list_keys:
			for i,image in enumerate(sample[list_key]):
				scale = 1/(np.median(image).clip(min=0.001))
				sample[list_key][i] = scale*image

		return sample


class Resize:
	def __init__(self,size,keys=[],list_keys=[]):
		self.keys = keys
		self.list_keys = list_keys
		self.size = size

	def __call__(self,sample):
		
		for key in self.keys:
			image = cv2.resize(sample[key].astype(np.float32),self.size, interpolation=cv2.INTER_LINEAR)
			if image.ndim == 2:
				image = np.expand_dims(image,2)
			sample[key] = image
					
		for list_key in self.list_keys:
			for i,image in enumerate(sample[list_key]):
				image = cv2.resize(image.astype(np.float32),self.size, interpolation=cv2.INTER_LINEAR)
				if image.ndim == 2:
					image = np.expand_dims(image,2)
				sample[list_key][i] = image
				
		return sample
		


