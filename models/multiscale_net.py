import torch
import data_processing.optical_flow_funs as OF
import matplotlib.pyplot as plt
import numpy as np
from models.resnet_module import *
from rendering.advanced_rendering_funs import eval_spot_light
from training_utils.loss_functions import masked_mean
from models.normal_integration_net import IntegationNet
from training_utils.make_scales import make_scales
from models.normal_pred_net import *




class MultiscaleNet(nn.Module):
	
	def __init__(self,use_mask=True, detach_scale=True, detach_integration=True, detach_light=True):
		super(MultiscaleNet, self).__init__()
		
		self.use_mask = use_mask
		self.detach_scale=detach_scale
		self.detach_integration=detach_integration
		self.detach_light=detach_light
		
		
		if use_mask:
			input_nc = 11
		else:
			input_nc = 10
		
		
		self.indiv_net_base =IndivNormalNet(input_nc,32)
		self.comb_net_base = CombinedNormalNet(3,32) 
		
		self.indiv_net_rec =IndivNormalNet(input_nc,32)
		self.comb_net_rec = CombinedNormalNet(3,32) 
		
		
		self.integration_net_base = IntegationNet(use_mask=use_mask)
		self.integration_net_rec = IntegationNet(use_mask=use_mask)
		
		
        	
	def normalize_images(self, images, mask):
		m = masked_mean(images[0],mask).view(-1,1,1,1)
		scale = 1/m.clamp(min=1e-8)

		normalized_images = []
		for im in images:
			normalized_images.append(im*scale)
		
		return normalized_images
		
        
        #list (b,3,m,n), (b,3,3), list light_data, (b,1,m,n), (b,3,m,n)
        #returns (b,3,m,n) (b,1,m,n)
	def main_forward(self, images, intrinsics, light_data, curr_depth, curr_normal, indiv_net_n, comb_net_n, integration_net, mask=None):
		#images list[N] (b,3,m,n)
		#intrinsics (b,3,3)
		#light_data see forward
		#curr_depth (b,1,m,n)
		#curr_normal (b,3,m,n)
		#returns (b,3,m,n) (b,1,m,n)
		
		#(b,m,n,3)
		ref_dirs = OF.get_camera_pixel_directions(images[0].shape[2:4],intrinsics,normalized_intrinsics=True)
		
		
		pc = curr_depth.squeeze(1).unsqueeze(3)*ref_dirs
	
		for i in range(0,len(images)):
			#(b,m,n,3) (b,m,n,1)
			n_to_light_vec, atten = eval_spot_light(pc, *light_data[i])
			
			n_to_light_vec = n_to_light_vec.permute(0,3,1,2)
			atten = atten.squeeze(3).unsqueeze(1)
			
			if not self.use_mask:
				indiv_net_in = torch.cat((images[i],n_to_light_vec,atten, curr_normal),dim=1)
			else:
				indiv_net_in = torch.cat((images[i],n_to_light_vec,atten, curr_normal,mask),dim=1)
				
			indiv_feature = indiv_net_n(indiv_net_in)
			if i == 0:
				combined = indiv_feature
			else:
				combined = torch.max(combined, indiv_feature)
			
		normal = comb_net_n(combined, curr_normal)
		normal = F.normalize(normal,dim=1)
		
		if self.detach_integration:
			normal_integ = normal.detach()
		else: 
			normal_integ = normal
			
		depth = integration_net(normal_integ, intrinsics, curr_depth, mask=mask)
		m = masked_mean(depth,mask).view(-1,1,1,1)
		depth = depth/m.clamp(min=1e-8)
		
		return normal, depth
        
	def make_scales_list(self, images):
		images_scales = []
		for im in images:
			images_scales.append( make_scales(im) )
		
		#transpose
		images_scales = list(map(list, zip(*images_scales)))
		return images_scales
		
	
	def forward(self, images, intrinsics, light_data, mask):
		#image list[N] (b,c,m,n)
		#intrinsics (b,3,3)
		#light_data list[N] (light_pos, light_dir, angular_attenuation) Where light_pos (b,3) , light_dir (b,3), angular_attenuation (b,) 
		device= images[0].device
		images = self.normalize_images(images,mask)
		images_scales = self.make_scales_list(images)
		
		if self.use_mask:
			mask_scales = make_scales(mask.float())
		else:
			mask_scales = len(images_scales)*[None]
			
		normal_out = []
		depth_out = []
		
			
		for i, (image_scale,mask_scale) in enumerate(zip(images_scales,mask_scales)):
			if i == 0:
				curr_normal = torch.zeros_like(images_scales[0][0])
				curr_normal[:,2,:,:] = -1
				s = curr_normal.shape
				curr_depth = torch.ones(s[0],1,s[2],s[3],device=device)
			
				curr_indiv_net = self.indiv_net_base
				curr_comb_net = self.comb_net_base
				curr_integ_net = self.integration_net_base
				
			else:
				curr_normal = F.interpolate(normal, scale_factor=2, mode='bilinear')
				curr_depth = F.interpolate(depth, scale_factor=2, mode='bilinear')
			
				curr_indiv_net = self.indiv_net_rec
				curr_comb_net = self.comb_net_rec
				curr_integ_net = self.integration_net_rec
			
			
			if self.detach_scale:
				curr_depth = curr_depth.detach()
				curr_normal = curr_normal.detach()
			
				
			normal, depth = self.main_forward(image_scale, intrinsics, light_data, curr_depth, curr_normal, curr_indiv_net, curr_comb_net, curr_integ_net, mask=mask_scale)
			
			normal_out.append(normal)
			depth_out.append(depth)
			
			
			
		
		outputs={'depth_scales': depth_out, 'normal_scales': normal_out}
		
			
		return outputs
	

		
