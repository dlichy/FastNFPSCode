import normal_integration.integrate_normals_perspective as integn
import normal_integration.near_functions as nf
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import torch
import data_processing.optical_flow_funs as OF


def perspective_normals_integrate_batch(normal, mask, intrinsics):
	#normal (b,3,m,n)
	#mask (b,1,m,n)
	#(b,3,3)
	#return (b,1,m,n)
	all_depths = []
	for b in range(normal.size(0)):
		depth = perspective_normals_integrate_single(normal[b:b+1,:,:,:],mask[b:b+1,:,:,:], intrinsics[b:b+1,:,:])
		all_depths.append(depth)
		
	return torch.cat(all_depths,dim=0)


def perspective_normals_integrate_single(normal, mask, intrinsics):
	#normal (1,3,m,n)
	#mask (1,1,m,n)
	#intrinsics (1,3,3)
	#return (1,1,m,n)
	l1_it=10 
	intrinsics = OF.normalized_intrinsics_to_pixel_intrinsics(intrinsics,normal.shape[2:4])
	intrinsics = intrinsics.squeeze().cpu().numpy()
	
	f = intrinsics[0,0]
	x0 = intrinsics[0,2]
	y0 = intrinsics[1,2]

	mask_img = mask[0].squeeze().cpu().numpy()
	mask_img = mask_img > 0.1
	normal = normal[0].permute(1,2,0).cpu().numpy()
	height= mask_img.shape[0]  
	width = mask_img.shape[1]   
    
 
	validsub = np.where(mask_img>0)
	mask_indx = validsub[0]*width + validsub[1]    

	(Gx,Gy,II,JJ,triangle_list)=integn.make_gradient(mask_img)
	u_f, v_f=nf.u_f_v_f(mask_indx,width,height,f,x0,y0,None,None)

	Z = np.ones(mask_indx.shape)
	
	Ngtm=np.zeros((Z.shape[0],3), dtype=float)
	for kk in range(3):
		nn=normal[:,:,kk]
		Ngtm[:,kk]=nn[mask_img>0]  
		
	mean_distance = 1
	Z = integn.integrate_normals_perspective(Ngtm, Z, f, u_f, v_f, mean_distance, l1_it,Gx,Gy,II,JJ)

	recon_depth = np.zeros(height*width)
	recon_depth[mask_indx] = Z
	recon_depth = recon_depth.reshape((height,width))

	recon_depth = torch.from_numpy(recon_depth).unsqueeze(0).unsqueeze(0).float()
	return recon_depth
	
