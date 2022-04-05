from datasets.io_nf import *
from rendering.advanced_rendering_funs import *
from rendering.torrance_bsdf import *
from rendering.uniform_sample_sphere_cap import *
from rendering.merl_brdf import *
import glob
import numpy as np

class RenderLikeLucesTorrance:
	def __init__(self, light_arg_generator):
	
		self.brdf_fun = TorranceBrdf()
		self.light_arg_generator = light_arg_generator
		
		
	def __call__(self, depth, normal, intrinsics, albedo, roughness):
		#depth (b,1,m,n)
		#normal (b,3,m,n)
		#intrinsics (b,3,3)
		#albedo (b,3,m,n)
		#roughness (b,1,m,n)
	
	
		#list (positon, direction, mu)
		light_args = self.light_arg_generator(depth.size(0),depth.device)
		
		brdf_arg_0 = albedo.permute(0,2,3,1)
		brdf_arg_1 = 0.05*torch.ones_like(depth).squeeze(1).unsqueeze(3)
		brdf_arg_2 = roughness.squeeze(1).unsqueeze(3)
		brdf_args = (brdf_arg_0, brdf_arg_1, brdf_arg_2)
		
		images = []
		
		for light_arg in light_args:
			image = render_point_like(depth, normal, intrinsics, eval_spot_light, light_arg, self.brdf_fun, brdf_args)
			images.append(image)
		
		return images, light_args




class GeneratedLightsRandom:
	def __init__(self, num_lights, mu_range=(0,2), pos_radius=1, pos_z_offset_range=(-0.2,0.2), cap_max_phi=np.pi/4):
		self.num_lights = num_lights
		self.mu_range = mu_range
		self.pos_radius = pos_radius
		self.pos_z_offset_range = pos_z_offset_range
		self.cap_max_phi = cap_max_phi
		
	
	def generate_pos(self, batch_size):
		#return (batch_size,3)
		#x and y are uniform on the circle
		r = self.pos_radius * torch.sqrt(torch.rand(batch_size))
		theta = 2*np.pi*torch.rand(batch_size)
		x = r*torch.cos(theta)
		y = r*torch.sin(theta)
		
		z = (self.pos_z_offset_range[1]-self.pos_z_offset_range[0])*torch.rand(batch_size) + self.pos_z_offset_range[0]
		
		pos = torch.stack((x,y,z),dim=1)
		return pos
		
	def generate_mu(self, batch_size):
		mu = (self.mu_range[1]-self.mu_range[0])*torch.rand(batch_size) + self.mu_range[0]
		return mu
		
	def generate_dir(self, batch_size):
		dir = uniform_sample_sphere_cap(self.cap_max_phi, batch_size)
		return torch.from_numpy(dir).float().t()
	
	
	def __call__(self, batch_size, device):
		#return list of tuples ((b,3), (b,3), (b,)) with length self.num_lights
		sample_light_data = []
		for _ in range(self.num_lights):
			
			pos = self.generate_pos(batch_size)
			dir = self.generate_dir(batch_size)
			mu = self.generate_mu(batch_size)
			
			
			sample_light_data.append( (pos.to(device), dir.to(device), mu.to(device)) )
			
		
		return sample_light_data



class GeneratedLightsRandomHemisphere:
	def __init__(self, num_lights, pos_z_offset_range=(-0.3,0.3)):
		self.num_lights = num_lights
		self.pos_z_offset_range = pos_z_offset_range
		
	
	def generate_pos_dir(self, batch_size):
		#return (b,3),(b,3)
		distance_from_origin = (self.pos_z_offset_range[1]-self.pos_z_offset_range[0])*torch.rand(batch_size) + self.pos_z_offset_range[0] + 1
		
		temp = torch.from_numpy(uniform_sample_sphere_cap(np.pi/2, batch_size)).float().view(-1,3)
		dir = temp
		pos_origin_centered = -distance_from_origin*temp
		pos_camera_centered = pos_origin_centered + torch.tensor([0,0,1]).view(1,3)
		
		return pos_camera_centered, dir
		

	
	#return list of tuples (b,3), (b,3), (b,) with length self.num_lights
	def __call__(self, batch_size, device):
		sample_light_data = []
		for _ in range(self.num_lights):
			
			pos,dir = self.generate_pos_dir(batch_size)
			mu = torch.zeros(batch_size)
			
			
			sample_light_data.append( (pos.to(device), dir.to(device), mu.to(device)) )
			
		
		return sample_light_data
		
class RenderLikeLucesMERL:
	def __init__(self, merl_path, light_arg_generator,device):
		merl_paths = glob.glob(os.path.join(merl_path,'*.binary'))
		print('found {} MERL files'.format(len(merl_paths)))
		self.brdf_fun = MutliMerlBrdf(merl_paths,device=device)
		self.light_arg_generator = light_arg_generator
		
		
	
	def __call__(self, depth, normal, intrinsics):
		#return list[self.num_lights] (b,3,m,n) 
	
	
		#list (positon, direction, mu)
		light_args = self.light_arg_generator(depth.size(0),depth.device)
		
		
		
		brdf_arg_0 = torch.randint(self.brdf_fun.num_brdfs,(depth.size(0),),device=depth.device)
		brdf_args = (brdf_arg_0,)
		
		images = []
		
		for light_arg in light_args:
			image = render_point_like(depth, normal, intrinsics, eval_spot_light, light_arg, self.brdf_fun, brdf_args)
			images.append(image)
		
		return images, light_args
		

class DataGenLikeLuces:
	def __init__(self, light_generator, merl_probability, merl_path,device):
		
		self.merl_probability = merl_probability
		self.torrance_renderer = RenderLikeLucesTorrance(light_generator)
		self.merl_renderer = RenderLikeLucesMERL(merl_path, light_generator,device=device)
		
		
	
	def __call__(self, sample):
		#sample needs normal, albedo, depth, intrinsics
		depth = sample['depth']
		normal = sample['normal']
		intrinsics = sample['intrinsics']
		
		if torch.rand(1) < self.merl_probability:
			images, light_args = self.merl_renderer(depth, normal, intrinsics)
		else:
			images, light_args = self.torrance_renderer(depth, normal, intrinsics, sample['albedo'], sample['roughness'])
			
		sample['images'] = images
		sample['light_data'] = light_args
		return sample











def plot_luces_light_pos():
	light_arg_generator = GeneratedLightsFromLuces(2,'/media/drive3/Downloads/Luces/Luces/data')
	light_data = light_arg_generator.light_data
	pos = light_data[:,0:3]
	dir = light_data[:,3:6]
	mu = light_data[:,6]
	
	angle = np.arccos(dir[:,2])
	plt.hist(angle)
	plt.show()
	
	z_offset = pos[:,2]
	plt.hist(z_offset)
	plt.show()
	
	plt.figure()
	ax = plt.axes(projection='3d')
	ax.scatter3D(pos[:,0],pos[:,1],pos[:,2])
	
	arrow_len = 0.2
	arrow_head = pos+arrow_len*dir
	ax.scatter3D(arrow_head[:,0], arrow_head[:,1], arrow_head[:,2])
	
	ax.scatter3D(0,0,0)
	ax.scatter3D(0,0,1)

	ax.set_xlim(-1,1)
	ax.set_ylim(-1,1)
	ax.set_zlim(-0.2,1)
	
	plt.show()
	
	plt.hist(mu)
	plt.show()

def plot_random_light_pos():
	light_arg_generator = GeneratedLightsRandom(2,pos_radius=0.6, cap_max_phi=np.pi/4)
	
	sample_light_data = light_arg_generator(50,'cpu')
	pos, dir, mu = sample_light_data[1]
	
	plt.figure()
	ax = plt.axes(projection='3d')
	ax.scatter3D(pos[:,0],pos[:,1],pos[:,2])
	
	arrow_len = 0.2
	arrow_head = pos+arrow_len*dir
	ax.scatter3D(arrow_head[:,0], arrow_head[:,1], arrow_head[:,2])
	
	ax.scatter3D(0,0,0)
	ax.scatter3D(0,0,1)
	ax.set_xlim(-1,1)
	ax.set_ylim(-1,1)
	ax.set_zlim(-0.2,1)
	
	plt.show()
	
	plt.hist(mu)
	plt.show()
	
def test_render_luces_like():
	light_arg_generator = GeneratedLightsFromLuces(2,'/media/drive3/Downloads/Luces/Luces/data')
	#light_arg_generator = GeneratedLightsRandom(2)
	renderer = RenderLikeLuces(light_arg_generator)

	training_transforms = transforms.Compose([
      ErodeMask(list_keys=['mask'],mask_erosion_size=(6,6)),
      NormalizeByMeanDepthMVPS(),
      MyToTensor(keys=[],list_keys=['image','albedo','mask','depth','normal','roughness'])
      ])


	dataset = SyntheticMVPS('/media/drive3/screen_ps/data_generation/dataset_MVPS/',TotallyRandomGetter(1), only_load_image_for_src=False, 		transform=training_transforms)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
	it = iter(dataloader)
	sample = it.next()
	#sample = it.next()
	print(sample.keys())
	
	
	albedo = sample['albedo'][0]
	roughness = sample['roughness'][0]
	normal = sample['normal'][0]
	intrinsics = sample['intrinsics'][0]
	depth = sample['depth'][0]
	
	images, light_args = renderer( depth, normal, intrinsics, albedo, roughness)

	b = 0
	plt.imshow(100*images[0][b].permute(1,2,0).cpu().numpy())
	plt.figure()
	plt.imshow(100*images[1][b].permute(1,2,0).cpu().numpy())
	plt.show()
	
	
def test_render_luces_merl():
	light_arg_generator = GeneratedLightsFromLuces(2,'/media/drive3/Downloads/Luces/Luces/data')
	#light_arg_generator = GeneratedLightsRandom(2)
	merl_path = '/media/drive2/MERL_DATA'
	renderer = RenderLikeLucesMERL(merl_path, light_arg_generator)

	training_transforms = transforms.Compose([
      ErodeMask(list_keys=['mask'],mask_erosion_size=(6,6)),
      NormalizeByMeanDepthMVPS(),
      MyToTensor(keys=[],list_keys=['image','albedo','mask','depth','normal','roughness'])
      ])


	dataset = SyntheticMVPS('/media/drive3/screen_ps/data_generation/dataset_MVPS/',TotallyRandomGetter(1), only_load_image_for_src=False, 		transform=training_transforms)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
	it = iter(dataloader)
	sample = it.next()
	#sample = it.next()
	print(sample.keys())
	
	
	albedo = sample['albedo'][0]
	roughness = sample['roughness'][0]
	normal = sample['normal'][0]
	intrinsics = sample['intrinsics'][0]
	depth = sample['depth'][0]
	
	images, light_args = renderer( depth, normal, intrinsics)

	b = 0
	plt.imshow(100*images[0][b].permute(1,2,0).cpu().numpy())
	plt.figure()
	plt.imshow(100*images[1][b].permute(1,2,0).cpu().numpy())
	plt.show()
	
if __name__ == '__main__':
	import matplotlib.pyplot as plt
	#test_render_luces_like()
	#plot_luces_light_pos()
	#plot_random_light_pos()
	test_render_luces_merl()
