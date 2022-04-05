from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import glob
import torch
import torchvision
from data_processing.exr_read_write import *
from data_processing.depth_to_depth_z import *
from datasets.synthetic_dataset_utils import *
import data_processing.optical_flow_funs as OF
import scipy.ndimage
import random
import json
import cv2
from collections import defaultdict


class SyntheticSculptureDataset(Dataset):

	def __init__(self, dataset_root, mode='train', domains_to_load = ['albedo','roughness','normal','depth','mask'], transform=None ):
		
		self.dataset_root = dataset_root
		self.mode = mode
		self.domains_to_load = domains_to_load
		self.transform = transform
		
	
		scene_list = sorted(glob.glob(os.path.join(dataset_root,'dataset/',mode,'*')))
		
		valid_scene_list = []
		
		for scene in scene_list:
			num_images = len( glob.glob( os.path.join(scene, '*.exr')) )
			if num_images != 180:
				continue
			valid_scene_list.append(scene)
			
		self.scene_list = valid_scene_list
		print('found %d complete scenes' % (len(self.scene_list)))


	
	def __len__(self):
		return len(self.scene_list)*36
	
	def __getitem__(self, idx):
		scene_idx = idx // 36
		scene_path = self.scene_list[scene_idx]
		xml_path = scene_path.replace('/dataset/','/xmls/')
		view_idx = idx % 36
		
		sample = {}
		sample['name'] = os.path.basename(scene_path)
		
		intrinsics = np.loadtxt(os.path.join(xml_path, 'intrinsics.txt'))
		n_intrinsics = OF.pixel_intrinsics_to_normalized_intrinsics(torch.from_numpy(intrinsics).unsqueeze(0).float(),(512,512)).squeeze()
		sample['intrinsics'] = n_intrinsics
		
		
		
		for k in self.domains_to_load:
			if k == 'mask':
				im_path = os.path.join(scene_path,'mask_%03d.exr' % view_idx)
				im = read_exr(im_path,channel_names=['Y']) > 0.01
			elif k == 'depth':
				im_path = os.path.join(scene_path,'depth_%03d.exr' % view_idx)
				depth = read_exr(im_path, channel_names=['Y'])
				depth = torch.from_numpy(depth).squeeze().unsqueeze(0).unsqueeze(0)
				im = depth_along_ray_to_depth_z(depth,n_intrinsics.unsqueeze(0)).squeeze().numpy()
			elif k == 'roughness':
				im_path = os.path.join(scene_path,'roughness_%03d.exr' % view_idx)
				im = read_exr(im_path, channel_names=['Y'])
			elif k == 'normal':
				im_path = os.path.join(scene_path, 'normal_%03d.exr' % view_idx)
				im = read_exr(im_path)
				im *= np.array([-1,-1,1.0]).reshape(1,1,3)
			else:
				im_path = os.path.join(scene_path, k + '_%03d.exr' % view_idx)
				im = read_exr(im_path)
			
			sample[k] = im
		

		if self.transform:
			self.transform(sample)
		
		return sample

