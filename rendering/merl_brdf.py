import numpy as np
import torch
import torch.nn.functional as F
import array
import struct

#d tuple, any tensor (b,d,k)
#return (b,k)
def ravel_idx(shape,x):
	p = list(shape)
	p.reverse()
	p = np.cumprod(p).tolist()[:-1]
	p.reverse()
	p.append(1)
	p = torch.tensor(p).unsqueeze(0).unsqueeze(2).to(x.device)
	idx = torch.sum(x*p,dim=1)
	return idx

def half_diff_to_MERL(theta_half, theta_diff, phi_diff):
	theta_half_idx = (90*torch.sqrt(theta_half.clamp(min=0.0)/(np.pi/2))).clamp(0,89).long()
	theta_diff_idx = (90*(theta_diff/(np.pi/2))).clamp(0,89).long()
	phi_diff_idx = (180*(phi_diff/np.pi) % 180).clamp(0,179).long()
	return theta_half_idx, theta_diff_idx, phi_diff_idx
	
def MERL_to_half_diff(theta_half_idx, theta_diff_idx, phi_diff_idx):
	theta_half = (np.pi/2) * ((theta_half_idx.float()+0.5)/90) ** 2  
	theta_diff = ((np.pi/2) * theta_diff_idx.float()/89).clamp(min=0.00001)
	phi_diff = (np.pi-0.000001) * (phi_diff_idx.float())/179 #needs to be less than pi for invertability purposes
	return theta_half, theta_diff, phi_diff
	
	
def read_merl(file_path):
	COLOR_SCALE = torch.tensor([1.0/1500.0,1.15/1500.0,1.66/1500.0]).view(3,1,1,1)
	with open(file_path, 'rb') as f:
		data = f.read()
		length = int(90*90*180)
		n = struct.unpack_from('3i', data)
		if  n[0]*n[1]*n[2] != length:
			raise IOError("Dimmensions doe not match")
		brdf = torch.from_numpy(np.frombuffer(data,dtype=np.float64,offset=struct.calcsize('3i'))).float()
		brdf = brdf.view(3,90,90,180)*COLOR_SCALE
	return brdf

def write_merl(brdf, save_path):
	COLOR_SCALE = torch.tensor([1.0/1500.0,1.15/1500.0,1.66/1500.0]).view(3,1,1,1)
	brdf = brdf/COLOR_SCALE
	with open(save_path, 'wb') as f:
		np.array([90,90,180]).astype(np.int32).tofile(f)
		brdf.numpy().astype(np.float64).tofile(f)
		
		
class MutliMerlBrdf:
	def __init__(self,merl_files,device='cpu'):
		self.merl_file_list = merl_files
		self.num_brdfs = len(merl_files)
		
		brdf_arrays = torch.zeros(len(merl_files),3,90,90,180)
		for i,f in enumerate(merl_files):
			brdf_arrays[i,:,:,:,:] = read_merl(f)
		
		self.brdf_arrays_r = brdf_arrays[:,0,...].reshape(-1).to(device)
		self.brdf_arrays_g = brdf_arrays[:,1,...].reshape(-1).to(device)
		self.brdf_arrays_b = brdf_arrays[:,2,...].reshape(-1).to(device)
		
		
		
	def __call__(self, L, V, N, index):
		#L (b,k,3)
		#V (b,k,3)
		#N (b,k,3)
		#index (b,)
		#return (b,k,3)
		s = L.shape
		L = L.view(s[0],-1,s[-1])
		V = V.view(s[0],-1,s[-1])
		N = N.view(s[0],-1,s[-1])


		L = F.normalize(L,dim=2)
		V = F.normalize(V,dim=2)
		N = F.normalize(N,dim=2)
	
		H = (L+V)/2
		H_n = F.normalize(H,dim=2)
		
		#(b,k)
		theta_half = torch.acos(torch.sum(N*H_n,dim=2))
		theta_diff = torch.acos(torch.sum(H_n*V,dim=2))
		N_cross_H = torch.cross(N,H,dim=2)
		
		temp1 = F.normalize(V-H,dim=2)
		temp2 = F.normalize(N_cross_H,dim=2)

		phi_diff = torch.acos(torch.sum(temp1*temp2,dim=2).clamp(min=-1,max=1)) + np.pi/2
		
	
		#(b,k)
		theta_half_idx, theta_diff_idx, phi_diff_idx = half_diff_to_MERL(theta_half, theta_diff, phi_diff)
		

		
		
		index = index.unsqueeze(1).expand_as(theta_half_idx)
		#(1,4,b*k)
		indices = torch.stack((index,theta_half_idx, theta_diff_idx, phi_diff_idx),dim=0).unsqueeze(0).view(1,4,-1)
		
		#(b*k)
		linear_indices = ravel_idx((self.num_brdfs,90,90,180),indices).squeeze(0)
			
		out_r = self.brdf_arrays_r[linear_indices].view(s[0],-1)
		out_g = self.brdf_arrays_g[linear_indices].view(s[0],-1)
		out_b = self.brdf_arrays_b[linear_indices].view(s[0],-1)
		
		cos_theta = torch.sum(N*L,dim=2,keepdim=True)
		
		out = torch.stack((out_r,out_g,out_b),dim=2)*cos_theta
		
		return out.view(s)


	
	
	
	
