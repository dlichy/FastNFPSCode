import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from training_utils.make_LHS import *
from models.resnet_module import ResnetBlock
from training_utils.make_LHS import make_left_hand_side
		
class RecNetComponent(nn.Module):
	def __init__(self, input_nc, init_feature_nc, output_nc, use_tanh=True, norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=True):
		super(RecNetComponent,self).__init__()
		
		self.use_tanh = use_tanh
		
		self.net = nn.Sequential(nn.Conv2d(input_nc, init_feature_nc, kernel_size=7, padding=3,bias=use_bias),
				 norm_layer(init_feature_nc),
				 nn.ReLU(True),
				 ResnetBlock(init_feature_nc,'zero',norm_layer,use_dropout,use_bias),
				 ResnetBlock(init_feature_nc,'zero',norm_layer,use_dropout,use_bias),
				 nn.Conv2d(init_feature_nc, 2*init_feature_nc, kernel_size=3,stride=2, padding=1, bias=use_bias),
				 norm_layer(2*init_feature_nc),
				 nn.ReLU(True),
				 ResnetBlock(2*init_feature_nc,'zero',norm_layer,use_dropout,use_bias),
				 ResnetBlock(2*init_feature_nc,'zero',norm_layer,use_dropout,use_bias),
                                nn.Conv2d(2*init_feature_nc, 4*init_feature_nc, kernel_size=3,stride=2, padding=1,bias=use_bias),
                                norm_layer(4*init_feature_nc),
				 nn.ReLU(True),
				 ResnetBlock(4*init_feature_nc,'zero',norm_layer,use_dropout,use_bias),
				 ResnetBlock(4*init_feature_nc,'zero',norm_layer,use_dropout,use_bias),
				 nn.Upsample(scale_factor=2,mode='bicubic'),
                                nn.Conv2d(4*init_feature_nc,2*init_feature_nc,kernel_size=3,padding=1),
			         norm_layer(2*init_feature_nc),
			         nn.ReLU(True),
			         ResnetBlock(2*init_feature_nc,'zero',norm_layer,use_dropout,use_bias),
				 ResnetBlock(2*init_feature_nc,'zero',norm_layer,use_dropout,use_bias),
				 nn.Upsample(scale_factor=2,mode='bicubic'),
                                nn.Conv2d(2*init_feature_nc,init_feature_nc,kernel_size=3,padding=1),
			         norm_layer(init_feature_nc),
			         nn.ReLU(True),
			         nn.Conv2d(init_feature_nc, output_nc, kernel_size=7, padding=3)
			         )
			         

	def forward(self, u_x, u_y, mask=None):
		dx = 2/u_x.size(3)
		dy = 2/u_y.size(2)
		if mask is not None:
			net_in = torch.cat((u_x*dx, u_y*dy, mask),dim=1)
		else:
			net_in = torch.cat((u_x*dx, u_y*dy),dim=1)
			
		net_out = self.net(net_in)
		if self.use_tanh:
			net_out = F.tanh(net_out)
			
		return net_out


class IntegationNet(nn.Module):
	def __init__(self, use_mask=False, use_tanh=True, init_feature_nc=32, norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=True):
		super(IntegationNet, self).__init__()
		if use_mask:
			self.net = RecNetComponent(3,init_feature_nc,1, use_tanh=use_tanh)
		else:
			self.net = RecNetComponent(2,init_feature_nc,1, use_tanh=use_tanh)
		
	def del_op(self, f):
		delta_x_i = 2/f.size(3)
		delta_y_i = 2/f.size(2)	
		f_x, f_y = image_derivativesLHS(f)
		f_x = f_x/delta_x_i
		f_y = f_y/delta_y_i
		return f_x, f_y
		
			         
	def main_forward(self, u, v, curr_log_depth, mask=None, gt_depth=None):
	
		f_x, f_y = self.del_op(curr_log_depth)
			
		u_bar = u - f_x
		v_bar = v - f_y
			
		g = self.net(u_bar, v_bar, mask=mask)
			
		log_depth = curr_log_depth + g
			
		return log_depth


	def forward(self, normal, intrinsics, curr_depth, mask=None, gt_depth=None):
		u, v  = make_left_hand_side(normal,intrinsics) #adjust for intrinsics
		curr_log_depth = torch.log(curr_depth.clamp(min=1e-8))
		log_depth = self.main_forward(u,v, curr_log_depth, gt_depth=gt_depth,mask=mask)
		depth = torch.exp(log_depth)
		return depth
			
			
			
		
		
		
		
		
		

