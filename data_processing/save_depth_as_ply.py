import torch
from data_processing.mask_triangulate import *
import data_processing.optical_flow_funs as OF
from data_processing.write_ply import *


def save_depth_as_ply(save_name, depth, mask, intrinsics_n):
	#save_name string
	#depth (1,1,m,n)
	#mask (1,1,m,n)
	#n_intrinsics = (1,3,3)
	dirs = OF.get_camera_pixel_directions(depth.shape[2:4],intrinsics_n).permute(0,3,1,2)
	pc = depth*dirs
	pc = pc.squeeze(0)[:,mask.squeeze()> 0.5]
	x,y,z = torch.unbind(pc,dim=0)
	
	mask = mask.squeeze().cpu().numpy() > 0.5
	faces, _, _ = mask_triangulate(mask)
	
	vert_props = [x.cpu().numpy(),y.cpu().numpy(),z.cpu().numpy()]
	
	write_ply(save_name,vert_props,prop_names=['x','y','z'],prop_types=['float32' for _ in range(0,3)],faces=faces)
