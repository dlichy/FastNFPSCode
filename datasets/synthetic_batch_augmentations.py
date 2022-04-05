import torch


class PatchMasker:
	def __init__(self, patch_size):
		self.patch_size = patch_size
		
	#list of (b,c,m,n)
	def __call__(self, images):
		for image in images:
			for im in image:
				top = torch.randint(im.size(1)-self.patch_size[0],(1,),device=im.device)
				left = torch.randint(im.size(2)-self.patch_size[1],(1,),device=im.device)
				im[:,top:(top+self.patch_size[0]),left:(left+self.patch_size[1])] = 0
	
class AddRandomNoise:
	def __init__(self, my_sigma):
		self.my_sigma = my_sigma
	
	#list of (b,c,m,n)
	def __call__(self,images):
		for image in images:
			image += self.my_sigma*torch.randn_like(image)



