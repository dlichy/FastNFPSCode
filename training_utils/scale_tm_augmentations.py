import torch
from training_utils.loss_functions import masked_mean
from data_processing.srgb_tonemapping import *

class ScaleAndTonemap:
	def __init__(self, scale_range=(0.05,0.2)):
		self.scale_range = scale_range
		
	#list of (b,c,m,n)
	def __call__(self, images, mask):
		#(b,)
		image_mean = masked_mean(images[0],mask)
		inv_mean = 1/image_mean.clamp(min=1e-8)
		rand_scale = (self.scale_range[1]-self.scale_range[0])*torch.rand_like(inv_mean) + self.scale_range[0]
		scale = inv_mean*rand_scale
		scale = scale.view(-1,1,1,1)
		new_images = []
		for im in images:
			new_images.append( linear_to_srgb(im*scale,clip=True) )
		return new_images


