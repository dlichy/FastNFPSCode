import torch
import numpy as np


class TorranceBrdf:
	def __init__(self,distribution='beckmann'):
		if distribution == 'beckmann':
			self.dist = self.beckmann
			self.smith_g1 = self.smith_g1_beckmann
		elif distribution == 'ggx':
			self.dist = self.ggx
			self.smith_g1 = self.smith_g1_ggx
		else:
			raise Exception('distribution not recognized')
		
		self.eps = 1e-8

	def __call__(self, L, V, N, kd, f0, alpha):
		#(b,...,3), (b,...,3), (b,...,3) (b,...,c) (b,...,1|c) (b,...,1)
		#return (b,c)
		s = L.shape
		L = L.reshape(-1,3)
		V = V.reshape(-1,3)
		N = N.reshape(-1,3)
		kd = kd.reshape(-1,3)
		f0 = f0.reshape(-1,1)
		alpha = alpha.reshape(-1,1)
		#(b,3)
		H = torch.nn.functional.normalize(L+V,dim=1)
		
		#(b,1)
		NH = torch.sum(N*H,dim=-1,keepdim=True)
		VN = torch.sum(V*N,dim=-1,keepdim=True)
		LN = torch.sum(L*N,dim=-1,keepdim=True)
		VH = torch.sum(V*H,dim=-1,keepdim=True)
		LH = torch.sum(L*H,dim=-1,keepdim=True)
		
		alpha2 = alpha**2
		#(b,1)
		D = self.dist(NH,alpha2)
		
		#(b,1)
		F = self.fresnel_schlick(f0, VH)
		
		#(b,1)
		G = self.smith_g1(LN,LH, alpha2)*self.smith_g1(VN, VH, alpha2)
		
		
		specular = F*D*G/(4*VN).clamp(min=self.eps)
		diff =  (kd/np.pi)*LN
		
		result = diff+specular
		result = result.clamp(min=0.0,max=10000.0)
		
		result = result.reshape(s)
		return result
		
	def eval_colocated(self, V, N, kd, f0, alpha):
		VN = torch.sum(V*N,dim=-1,keepdim=True)
		alpha2 = alpha**2
		D = self.dist(VN,alpha2)
		G = self.smith_g1(VN,1,alpha2)**2
		
		specular = f0*D*G/(4*VN).clamp(min=self.eps)
		
		diff =  (kd/np.pi)*VN
		
		
		result = diff+specular
		result = result.clamp(min=0.0,max=10000.0)
		
		return result
		
	
	def fresnel_schlick(self, f0, VH):
		#(b,1), (b,1)
		#return (b,1)
		return f0 + (1.0 - f0) * torch.pow(1.0 - VH, 5.0)
		

	def beckmann(self, NH, alpha2):
		#(b,1), (b,1)
		#return (b,1)
		cos_half2 = NH**2
		sin_half2 = 1-cos_half2
		tan_half2 = sin_half2/cos_half2.clamp(min=self.eps)
		num = torch.exp(-tan_half2/alpha2.clamp(min=self.eps))
		denom = np.pi*alpha2*cos_half2**2
		return num/denom.clamp(min=self.eps)
		
	def ggx(self, NH, alpha2):
		cos_half2 = NH**2
		sin_half2 = 1-cos_half2
		denom = np.pi*(sin_half2+alpha2*cos_half2)**2
		return alpha2/denom.clamp(min=self.eps)
	
	
	def smith_g1_beckmann(self, XN, XH, alpha2):
		#(b,1), (b,1), (b,1)
		#return (b,1)
		xy_alpha2 = alpha2*(1-XN**2)
		tan_theta_alpha2 = xy_alpha2/ ( (XN**2).clamp(min=self.eps) )
		
		a = 1/torch.sqrt(tan_theta_alpha2.clamp(min=self.eps))
		a_sqr = a**2
		
		mask = a > 1.6
		result = (3.535 * a + 2.181 * a_sqr) / (1 + 2.276 * a + 2.577 * a_sqr)
		result[mask] = 1.0
		
		result[ XH*XN <= 0] = 0
		
		return result
	
	def smith_g1_ggx(self, XN, XH, alpha2):
		xy_alpha2 = alpha2*(1-XN**2)
		tan_theta_alpha2 = xy_alpha2/ ( (XN**2).clamp(min=self.eps) )
		result = 2.0 / (1.0 + torch.sqrt((1 + tan_theta_alpha2).clamp(min=self.eps)))
		result[ XH*XN <= 0] = 0
		return result
		
		
		
		
		
		

	
	
		
		
		
		
