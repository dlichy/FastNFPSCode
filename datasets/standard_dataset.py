from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
import glob
import torch
import data_processing.optical_flow_funs as OF
import scipy.ndimage
import cv2


class StandardDataset(Dataset):

	def __init__(self,dataset_root, inv_tonemap, transform=None):
		self.transform = transform
		self.inv_tonemap = inv_tonemap
		self.dataset_root = dataset_root
		self.scene_dirs = glob.glob(os.path.join(dataset_root,'*'))
		self.scene_dirs = sorted([x for x in self.scene_dirs if os.path.isdir(x)])
		
		print('Found %d real images' % len(self.scene_dirs))
		
		
	def __len__(self):
		return len(self.scene_dirs)


	def load_image(self,image_path):
		
		cv2_im = cv2.imread(image_path, -1)
		cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)            
		if cv2_im.dtype=='uint16':
			cv2_im=np.float32(cv2_im)/65535.0
		else:
			cv2_im=np.float32(cv2_im)/255.0
		return cv2_im

	def __getitem__(self, idx):
		scene_dir = self.scene_dirs[idx]
		
		sample = {}
		sample['name'] = os.path.basename(scene_dir)
		sample['idx'] = idx
		
		image_paths = sorted(glob.glob(os.path.join(scene_dir,'*.jpg')))
		

		images = []
		for im_path in image_paths:
			image = self.load_image(im_path)
			if self.inv_tonemap:
				image = image**2.2
			images.append(image)
		sample['images'] = images
		
		mask = cv2.imread(os.path.join(scene_dir,'mask.png'))
		
		mask = mask[:,:,0:1]
		sample['mask'] = mask > 0.5
		
		
		s = mask.shape
	
		intrinsics=np.loadtxt(os.path.join(scene_dir,'intrinsics.txt'))

		n_intrinsics = OF.pixel_intrinsics_to_normalized_intrinsics(torch.from_numpy(intrinsics).unsqueeze(0).float(),mask.squeeze().shape).squeeze(0)

		sample['intrinsics'] = n_intrinsics
		sample['mean_distance'] = 1.0

		if self.transform:
			sample = self.transform(sample)
			

		return sample
		

	
	
	
	
	
	

