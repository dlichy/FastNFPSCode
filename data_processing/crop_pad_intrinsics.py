import torch
import numpy as np
import data_processing.optical_flow_funs as OF


def crop_intrinsics(top_bottom, left_right, intrinsics):
	intrinsics_out = intrinsics.clone()
	intrinsics_out[0,2] -= left_right[0]
	intrinsics_out[1,2] -= top_bottom[0]
	return intrinsics_out
	
def crop_intrinsics_normalized(top_bottom, left_right, n_intrinsics, image_shape):
	intrinsics = OF.normalized_intrinsics_to_pixel_intrinsics(n_intrinsics.unsqueeze(0), image_shape).squeeze(0)
	new_intrinsics = crop_intrinsics(top_bottom, left_right, intrinsics)
	new_image_shape = (top_bottom[1]-top_bottom[0],left_right[1]-left_right[0])
	new_intrinsics_n = OF.pixel_intrinsics_to_normalized_intrinsics(new_intrinsics.unsqueeze(0),new_image_shape).squeeze(0)
	return new_intrinsics_n
	
def crop_image(top_bottom, left_right, image, channel_last = True):
	if channel_last:
		image_shape = image.shape[0:2]
	else:
		image_shape = image.shape[1:3]
		
	pad_needed, top_bottom_pad, left_right_pad = calculate_need_padding(top_bottom,left_right,image_shape)
	if pad_needed:
		image = pad_image(top_bottom_pad, left_right_pad, image, channel_last = channel_last)
		left_right = (left_right[0]+left_right_pad[0],left_right[1]+left_right_pad[0])
		top_bottom = (top_bottom[0]+top_bottom_pad[0],top_bottom[1]+top_bottom_pad[0])
		
	if channel_last:
		image_out = image[top_bottom[0]:top_bottom[1],left_right[0]:left_right[1],...]
	else:
		image_out = image[:,top_bottom[0]:top_bottom[1],left_right[0]:left_right[1]]
		
	return image_out
	
def calculate_need_padding(top_bottom,left_right,image_shape):
	if top_bottom[0] < 0:
		top_pad = -top_bottom[0]
	else: 
		top_pad = 0
		
	if left_right[0] < 0:
		left_pad = -left_right[0]
	else: 
		left_pad = 0
		
	if top_bottom[1] > image_shape[0]:
		bottom_pad = top_bottom[1]-image_shape[0]
	else: 
		bottom_pad = 0
		
	if left_right[1] > image_shape[1]:
		right_pad = left_right[1] - image_shape[1]
	else: 
		right_pad = 0
	
	top_bottom_pad = (top_pad,bottom_pad)
	left_right_pad = (left_pad,right_pad)
	
	pad_needed = (top_pad != 0) or (bottom_pad != 0) or (left_pad != 0) or (right_pad != 0)
	
	return pad_needed, top_bottom_pad, left_right_pad
		
		
	
	
def pad_image(top_bottom, left_right, image, channel_last = True):
	is_numpy = type(image) is np.ndarray
	if is_numpy:
		image = torch.from_numpy(image)

	if channel_last:
			image_out = torch.nn.functional.pad(image, (0,0,left_right[0],left_right[1], top_bottom[0],top_bottom[1]))
	else:
			image_out = torch.nn.functional.pad(image, (left_right[0],left_right[1], top_bottom[0],top_bottom[1]))
		
	if is_numpy:
		image_out = image_out.numpy()
	return image_out
	
	

def pad_intrinsics(top_bottom_pad, left_right_pad, intrinsics):
	top_bottom_crop = (-top_bottom_pad[0],0)
	left_right_crop = (-left_right_pad[0],0)
	return crop_intrinsics(top_bottom_crop, left_right_crop, intrinsics)
	

def pad_intrinsics_normalized(top_bottom, left_right, n_intrinsics, image_shape):
	intrinsics = OF.normalized_intrinsics_to_pixel_intrinsics(n_intrinsics.unsqueeze(0), image_shape).squeeze(0)
	new_intrinsics = pad_intrinsics(top_bottom, left_right, intrinsics)
	new_image_shape = (image_shape[0]+top_bottom[1]+top_bottom[0],image_shape[1]+left_right[1]+left_right[0])
	new_intrinsics_n = OF.pixel_intrinsics_to_normalized_intrinsics(new_intrinsics.unsqueeze(0),new_image_shape).squeeze(0)
	return new_intrinsics_n
	

	
