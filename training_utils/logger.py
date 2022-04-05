from collections import defaultdict
from data_processing.save_depth_as_ply import *
import os
from PIL import Image
import torch
import numpy as np
from torchvision import transforms

class ScalarLogger:
	
	def __init__(self, save_file, log_freq=50, keys=[]):
		
		self.counter = 0
		self.save_file = save_file
		self.log_freq = log_freq
		self.keys = keys
		
		#dict with keys=scalar_keys and values = sum of losses until that point
		self.log_since_last = defaultdict(lambda: 0)
		#dict with keys=scalar_keys and values = sum of losses until that point
		self.log_all = defaultdict(lambda: 0)
		
		self.log_string = ''
			
	#data is a dict where values are floats or list of floats
	def __call__(self, data, other_text=''):
		for key in self.keys:
			if not isinstance(data[key], list):
				self.log_all[key] += data[key]
				self.log_since_last[key] += data[key]
			else:
				for i,v in enumerate(data[key]):
					self.log_all[key + ('_%d'%i)] += v
					self.log_since_last[key + ('_%d'%i)] += v
		self.counter +=1
		if self.counter % self.log_freq == 0:
			string = 'iter {} - {} \n'.format(self.counter-self.log_freq,self.counter-1)
			for k, v in sorted(self.log_since_last.items()):
				string += '{}: {} \n'.format(str(k), str(v/self.log_freq))
			string += 'other text: {} \n'.format(other_text)
			print(string)
			self.log_string += string
			self.log_since_last = defaultdict(lambda: 0)
    		    	
	def summarize(self):
		string = '----------  Epoch Summary -----------\n'
		for k, v in sorted(self.log_all.items()):
			string += '{}: {} \n'.format(str(k), str(v/self.counter))
		print(string)
		self.log_string += string
		with open(self.save_file,'w') as f:
			f.write(self.log_string)
			

class NormalScale:
	def __init__(self):
		self.flip = torch.tensor([1,-1,-1.0]).view(1,3,1,1)
		
	def __call__(self,sample):
		for key in sample.keys():
			if 'normal' in key:
				if not isinstance(sample[key], list):
					im = sample[key]
					sample[key] = (im*self.flip.to(im.device)+1)/2
				else: 
					for i,im in enumerate(sample[key]):
						sample[key][i] = (im*self.flip.to(im.device)+1)/2
		return sample
		
class DepthScale:
	def __init__(self,min=0.7,max=1.3):
		self.min = torch.tensor(min).view(1,1,1,1)
		self.max = torch.tensor(max).view(1,1,1,1)
		
	def __call__(self,sample):
		for key in sample.keys():
			if 'depth' in key:
				if not isinstance(sample[key], list):
					im = sample[key]
					sample[key] = (im.to('cpu') - self.min)/(self.max-self.min)
				else: 
					for i,im in enumerate(sample[key]):
						sample[key][i] = (im.to('cpu') - self.min)/(self.max-self.min)
		return sample


class ImageLogger:
	# keys = None saves all images in the dict
	# indices_to_save = a list of indices in the batch to save. None saves all images in the batch
	def __init__(self,save_root, keys=None, log_freq=50, transform=transforms.Compose([NormalScale(),DepthScale()]),save_first_batch_only=True):
		if not os.path.exists(save_root):
			os.mkdir(save_root)
			
		self.counter = 0
		self.keys = keys
		self.log_freq = log_freq
		self.save_root = save_root
		self.transform = transform
		self.save_first_batch_only = save_first_batch_only
		
	
	#image is tensor of size (1|3,m,n)
	def save_image(self, save_name, image):
		save_name = save_name + '.png'
		image = image.permute([1,2,0]).detach().cpu().numpy()
		pil_image = Image.fromarray(np.uint8(255*image.squeeze().clip(0,1)))
		pil_image.save(save_name)
		
	def save_image_batch(self, save_name, image):
		#determine which images of the batch to save
		if self.save_first_batch_only:
			indices = [0]
		else:
			indices = range(image.size(0))
			
		for b in indices:
			self.save_image(save_name + 'b{}'.format(b), image[b,:,:,:])
				
		
	
	#values for image_dict are either tensors of shape (b,1|3,m,n) or lists of (b,1|3,m,n)
	#list of folder names to save images under
	def __call__(self, image_dict, folder_name, mesh_data=None):
		self.counter += 1
		if self.counter % self.log_freq == 0:
			folder_name = os.path.join(self.save_root, folder_name)
			if not os.path.exists(folder_name):
				os.mkdir(folder_name)
		
			if self.transform is not None:
				image_dict = self.transform(image_dict)
			
			if self.keys is not None:
				curr_keys = self.keys
			else:
				curr_keys = image_dict.keys()

			for key in curr_keys:
				if not isinstance(image_dict[key], list):
					self.save_image_batch(os.path.join(folder_name,key),image_dict[key])
				else:
					for i,v in enumerate(image_dict[key]):
						self.save_image_batch(os.path.join(folder_name,key + ('_s%d_'%i)),v)
		
			if mesh_data is not None:
				for b in range(0,mesh_data['net_depth'].size(0)):
				
					save_depth_as_ply(os.path.join(folder_name,'net_depth_mesh_b%d.ply' % b), mesh_data['net_depth'][b:b+1,...], mesh_data['mask'][b:b+1,...], mesh_data['intrinsics'])
					if 'int_depth' in mesh_data:
						save_depth_as_ply(os.path.join(folder_name,'int_depth_mesh_b%d.ply' % b), mesh_data['int_depth'][b:b+1,...], mesh_data['mask'][b:b+1,...], mesh_data['intrinsics'])
		
					
		
			
	
	
