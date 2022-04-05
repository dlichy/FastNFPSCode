import torch
import matplotlib.pyplot as plt
import numpy as np
from training_utils.loss_functions import masked_mean
from data_processing.resample_by_intrinsics import *
from models.LCNet import *
import data_processing.optical_flow_funs as OF


class CalibrationNet(nn.Module):
	
	def __init__(self, batch_norm=False):
		super(CalibrationNet, self).__init__()
		self.feature_net = FeatExtractor(batch_norm,3)
		self.regression_net = Classifier(batch_norm, 256)
		self.intrinsics = torch.tensor([[[2.7475e+00, 0.0000e+00, 1.9531e-03],
         [0.0000e+00, 2.7475e+00, 1.9531e-03],
         [0.0000e+00, 0.0000e+00, 1.0000e+00]]])
		
	
        
	def normalize_images(self, images, mask):
		m = masked_mean(images[0],mask).view(-1,1,1,1)
		scale = 1/m.clamp(min=1e-8)

		normalized_images = []
		for im in images:
			normalized_images.append(im*scale)
		
		return normalized_images
        
	def resample(self,images,mask,intrinsics):
        	device = mask.device
        	b = mask.size(0)
        	mask_resamp = resample_by_intrinsics(mask.float(), intrinsics, self.intrinsics.expand(b,-1,-1).to(device), (128,128))
        	
        	images_resamp = []
        	for im in images:
        		images_resamp.append( resample_by_intrinsics(im, intrinsics, self.intrinsics.expand(b,-1,-1).to(device), (128,128)) )
        	

        
        	return images_resamp, mask_resamp
        

	def forward(self, images, mask, intrinsics):
		#images list[N] (b,3,m,n)
		#mask list[N] (b,1,m,n)
        	#return list[N] (b,3)
		images = self.normalize_images(images,mask)
		images, mask = self.resample(images, mask, intrinsics)
		
		
		features = [self.feature_net(x) for x in images]
		combined_feature, _ = torch.max(torch.stack(features,dim=0),dim=0)
		
		light_pos = []
		for f in features:
			lp = self.regression_net(torch.cat((f,combined_feature),dim=1))
			light_pos.append(lp)
		
		return light_pos
        
        
        
	

		
