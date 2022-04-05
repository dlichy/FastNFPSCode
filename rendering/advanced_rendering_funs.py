import torch
import torch.nn.functional as F
import numpy as np
import data_processing.optical_flow_funs as OF
from rendering.torrance_bsdf import *
import matplotlib.pyplot as plt




def eval_dir_light(pos, light_dir):
	#pos (b,m,n,3)
	#light_dir (b,3)
	#return (b,m,n,3) (b,m,n,1)
	s = pos.shape
	to_light_vec = -light_dir.unsqueeze(1).unsqueeze(1).expand(-1,s[1],s[2],-1)
	atten = torch.ones(s[0],s[1],s[2],1)
	return to_light_vec, atten
	

def eval_point_light(pos, light_pos):
	#pos (b,m,n,3)
	#light_pos (b,3)
	#return (b,m,n,3) (b,m,n,1)
	to_light_vec = light_pos.unsqueeze(1).unsqueeze(1) - pos
	n_to_light_vec = F.normalize(to_light_vec,dim=3)
	#(b,m,n,1)
	len_to_light_vec = torch.norm(to_light_vec,dim=3,keepdim=True)
	atten = 1/(len_to_light_vec**2).clamp(min=1e-8)
	return n_to_light_vec, atten


def eval_spot_light(pos, light_pos, light_dir, mu):
	#pos (b,m,n,3)
	#light_pos (b,3)
	#light_dir (b,3)
	#mu (b,)
	#return (b,m,n,3) (b,m,n,1)
	
	to_light_vec = light_pos.unsqueeze(1).unsqueeze(1) - pos
	n_to_light_vec = F.normalize(to_light_vec,dim=3)
	#(b,m,n,1)
	len_to_light_vec = torch.norm(to_light_vec,dim=3,keepdim=True)
	light_dir_dot_to_light = torch.sum(-n_to_light_vec*light_dir.unsqueeze(1).unsqueeze(1),dim=3,keepdim=True).clamp(min=1e-8)
	numer = torch.pow(light_dir_dot_to_light, mu.view(-1,1,1,1))
	atten = numer/(len_to_light_vec**2).clamp(min=1e-8)
	return n_to_light_vec, atten


def eval_diffuse_brdf(L, V, N, albedo):
	LN = torch.sum(L*N,dim=-1,keepdim=True).clamp(min=0.0)
	diff =  (albedo/np.pi)*LN
	return diff



def render_point_like(depth, normal, intrinsics, eval_light_fun, eval_light_args, brdf_fun, brdf_args, depth_is_along_ray=False):
	
	#(b,m,n,3)
	dirs = OF.get_camera_pixel_directions(normal.shape[2:4],intrinsics,normalized_intrinsics=True)
	if depth_is_along_ray:
		dirs = F.normalize(dirs,dim=3)
	#(b,m,n,3)
	pos = dirs*depth.squeeze(1).unsqueeze(3)
	
	to_light_vec, atten = eval_light_fun(pos,*eval_light_args)
	
	brdf = brdf_fun(to_light_vec, -dirs, normal.permute(0,2,3,1), *brdf_args)
	
	
	image = (brdf*atten).permute(0,3,1,2)
	return image
	

	

