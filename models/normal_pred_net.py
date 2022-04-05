import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from models.resnet_module import ResnetBlock

class IndivNormalNet(nn.Module):
	def __init__(self,input_ch=3, init_feature_nc=32, norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=True):
		super(IndivNormalNet,self).__init__()
		
		self.net = nn.Sequential(nn.Conv2d(input_ch, init_feature_nc, kernel_size=7, padding=3,bias=use_bias),
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
				 nn.Conv2d(4*init_feature_nc, 4*init_feature_nc, kernel_size=3, padding=1, bias=use_bias)
				 )

	def forward(self, image):
		return self.net(image)
		
		
class CombinedNormalNet(nn.Module):
	def __init__(self,output_ch, init_feature_nc=32,  norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=True):
		super(CombinedNormalNet,self).__init__()
		self.net = nn.Sequential( ResnetBlock(4*init_feature_nc,'zero',norm_layer,use_dropout,use_bias),
                                nn.ConvTranspose2d(4*init_feature_nc,2*init_feature_nc,kernel_size=3, stride=2,padding=1, output_padding=1,bias=use_bias),
			         norm_layer(2*init_feature_nc),
			         nn.ReLU(True),
			         ResnetBlock(2*init_feature_nc,'zero',norm_layer,use_dropout,use_bias),
				 ResnetBlock(2*init_feature_nc,'zero',norm_layer,use_dropout,use_bias),
				 nn.ConvTranspose2d(2*init_feature_nc,init_feature_nc,kernel_size=3, stride=2,padding=1, output_padding=1,bias=use_bias),
			         norm_layer(init_feature_nc),
			         nn.ReLU(True),
			         
			         )
		self.regular_out = nn.Conv2d(init_feature_nc, output_ch, kernel_size=7, padding=3)
		self.scale_factor_out = nn.Conv2d(init_feature_nc, 1, kernel_size=7, padding=3)
		
    
	def forward(self, x, init):
		x = self.net(x)
		out = self.regular_out(x) + init
		return out
    
    
    
    
    
    
    
    
    
    
