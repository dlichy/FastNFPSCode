from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
import glob
import torch
import matplotlib.pyplot as plt
from datasets.io_nf import *
import data_processing.optical_flow_funs as OF
import scipy.ndimage
import random
import cv2
from collections import defaultdict


class LucesDataset(Dataset):

	def __init__(self,dataset_root, raw=True, initial_resize=None, lights_to_load=range(0,52),ignoreBoundaries=False, transform=None):
		self.transform = transform
		self.ignoreBoundaries = ignoreBoundaries
		self.dataset_root = dataset_root
		self.scene_dirs = glob.glob(os.path.join(dataset_root,'*'))
		self.scene_dirs = sorted([x for x in self.scene_dirs if os.path.isdir(x)])
		self.lights_to_load=lights_to_load
		self.raw=raw
		self.initial_resize=initial_resize
		
		
		print('Found %d LUCES scenes' % len(self.scene_dirs))
		
		
	def __len__(self):
		return len(self.scene_dirs)


	def load_image(self,scene_path, light_num, raw):
		if raw:
			image_path = os.path.join(scene_path,'RAW','%02d.png' % (light_num+1))
		else:
			image_path = os.path.join(scene_path,'%02d.png' % (light_num+1))
		cv2_im = cv2.imread(image_path, -1)
		if not raw:
			cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)            
		if cv2_im.dtype=='uint16':
			cv2_im=np.float32(cv2_im)/65535.0
		else:
			cv2_im=np.float32(cv2_im)/255.0
		if self.initial_resize is not None:
			cv2_im = cv2.resize(cv2_im,self.initial_resize)
			
		return cv2_im

	def __getitem__(self, idx):
		scene_dir = self.scene_dirs[idx]
		
		if self.raw:
			setup_file = os.path.join(scene_dir,'RAW','led_params.txt')
			normal_path = os.path.join(scene_dir,'RAW','normals.png')
			depth_path = os.path.join(scene_dir,'RAW','zgt.npz')
			mask_path = os.path.join(scene_dir,'RAW','mask.png')
		else:
			setup_file = os.path.join(scene_dir,'led_params.txt')
			normal_path = os.path.join(scene_dir,'normals.png')
			depth_path = os.path.join(scene_dir,'zgt.npz')
			mask_path = os.path.join(scene_dir,'mask.png')
			
		N_img, ncols, nrows, f, x0, y0, mean_distance, Lpos, Ldir, Phi, mu = readNFSetup(setup_file)
		
		sample = {}
		light_idx_to_load = np.array(self.lights_to_load)
		sample['name'] = os.path.basename(scene_dir)
		light_pos = [Lpos[i,:] for i in self.lights_to_load]
		light_dir = torch.unbind(torch.from_numpy(Ldir[light_idx_to_load,:]),0)
		mu = torch.unbind(torch.from_numpy(mu[light_idx_to_load,:]),0)
		sample['light_data'] = [x for x in zip(light_pos, light_dir, mu)]
		
		sample['phi'] = torch.unbind(torch.from_numpy(Phi[light_idx_to_load,:]),0)
		sample['mean_distance'] = np.array(mean_distance,dtype=np.float32)
		intrinsics = np.eye(3)
		intrinsics[0,0] = f
		intrinsics[1,1] = f
		intrinsics[0,2] = x0
		intrinsics[1,2] = y0
		
		
		nShape,mask, validind  = load_mask_indx(mask_path, ignoreBoundaries=self.ignoreBoundaries,scale=1)
		normal = load_normal_map(normal_path)
		depth = read_gt_depth(depth_path)
		
		n_intrinsics = OF.pixel_intrinsics_to_normalized_intrinsics(torch.from_numpy(intrinsics).unsqueeze(0).float(),(1536,2048)).squeeze(0)
		sample['intrinsics'] = n_intrinsics
		
		
		images = []
		for light_num in self.lights_to_load:
			image = self.load_image(scene_dir,light_num,raw=self.raw)
			images.append(image)
		

		
		sample['images'] = images
		sample['normal'] = normal
		sample['mask'] = mask > 0.1
		sample['depth'] = depth	
				
		if self.transform:
			sample = self.transform(sample)
		
		return sample
		
class ScaleByPhi():
	def __call__(self, sample):
		for i in range(len(sample['images'])):
			sample['images'][i] /= sample['phi'][i].view(1,1,3).numpy()
		return sample
		
class NormalizeByMeanDepthLuces():
	def __call__(self, sample):
		mean_depth = sample['mean_distance']
		scale = 1/mean_depth
		sample['depth'] *= scale
		
		new_light_data = []
		for light_datum in sample['light_data']:
			new_light_datum = (light_datum[0]*scale, light_datum[1], light_datum[2])
			new_light_data.append(new_light_datum)
			
		sample['light_data'] = new_light_data
		return sample
	
	
	
	
	
	

