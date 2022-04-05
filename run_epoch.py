import torch
from training_utils.make_scales import make_scales
from data_processing.depth_to_normals import get_normals_from_depth_list
from normal_integration.integrate_normals_adaptor import perspective_normals_integrate_batch
import matplotlib.pyplot as plt

def prepare_gt(sample,device):
	gt = {}
	gt['mask_scales'] = make_scales(sample['mask'].float())
	if 'depth' in sample:
		gt['depth_scales'] = make_scales(sample['depth'])
		gt['nfd_scales'] = get_normals_from_depth_list(gt['depth_scales'],sample['intrinsics'])
	
	if 'normal' in sample:
		gt['normal_scales'] = make_scales(sample['normal'])
	
	gt['intrinsics'] = sample['intrinsics']
	gt['mean_distance'] = sample['mean_distance']
	return gt


class PreprocessSynthetic:
	def __init__(self, renderer, device, image_augmentations=[]):
		self.renderer = renderer
		self.device = device
		self.image_augmentations = image_augmentations
		
	def __call__(self, sample):
		for k in ['depth', 'normal', 'albedo', 'roughness', 'intrinsics','mask','mean_distance']:
			sample[k] = sample[k].to(self.device)
	
		sample = self.renderer(sample)
		for aug in self.image_augmentations:
			aug(sample['images'])
		
		return sample
		
class PreprocessReal:
	def __init__(self, device):
		self.device = device
		
	def __call__(self, sample):
		
		for i in range(len(sample['images'])):
			sample['images'][i] = sample['images'][i].to(self.device)
		
		for k in ['depth', 'normal','intrinsics','mean_distance','mask']:
			if k in sample.keys():
				sample[k] = sample[k].to(self.device)
		
		return sample
		
def train_epoch(net, dataloader, sample_preprocessing_fun, device, criterion, optimizer, scalar_logger=None, image_logger=None):
	
	net.train()

	for batch_num, sample in enumerate(dataloader):
		
		sample = sample_preprocessing_fun(sample)
		target = prepare_gt(sample,device)


		optimizer.zero_grad()
		
		output = net(sample['images'],sample['intrinsics'],sample['light_data'],sample['mask'])
		
		loss,losses_dict = criterion(output,target)
		
		loss.backward()
		optimizer.step()
			
		
		if scalar_logger is not None:
			scalar_logger(losses_dict,sample['name'][0])
			
		if image_logger is not None:
			log_images = {}
			for k,v in output.items():
				log_images['output_' + k] = v
			for k,v in target.items():
				log_images['target_' + k] = v 
			log_images['images'] = sample['images']
			image_logger(log_images,sample['name'][0])
			
def eval_epoch(net, dataloader, sample_preprocessing_fun, device, criterion=None,  scalar_logger=None, image_logger=None, calibration_net=None, post_integrate_normals=False, log_meshes=False):
			
	net.eval()
	if calibration_net is not None:
		calibration_net.eval()
	
	for batch_num, sample in enumerate(dataloader):
		
		sample = sample_preprocessing_fun(sample)
		target = prepare_gt(sample,device)
		
		if calibration_net is not None:
			light_pos = calibration_net(sample['images'],sample['mask'],sample['intrinsics'])
			light_dir = torch.tensor([0,0,1.0],device=device).view(1,3)
			mu = torch.tensor([0.0],device=device)
			new_light_data = [(lp,light_dir,mu) for lp in light_pos]
			sample['light_data'] = new_light_data

		output = net(sample['images'],sample['intrinsics'],sample['light_data'],sample['mask'])
		
		if post_integrate_normals:
			int_depth = perspective_normals_integrate_batch(output['normal_scales'][-1], sample['mask'], sample['intrinsics'])
			output['int_depth'] = int_depth
		
		if criterion is not None:
			loss,losses_dict = criterion(output,target)
		
		if scalar_logger is not None:
			scalar_logger(losses_dict,sample['name'][0])
			
		if image_logger is not None:
			log_images = {}
			for k,v in output.items():
				log_images['output_' + k] = v
			for k,v in target.items():
				log_images['target_' + k] = v 
			log_images['images'] = sample['images']
			if log_meshes:
				mesh_data= {}
				mesh_data['mask'] = sample['mask'].to('cpu')
				mesh_data['intrinsics'] = sample['intrinsics'].to('cpu')
				mesh_data['net_depth'] = output['depth_scales'][-1].to('cpu')
				if post_integrate_normals:
					mesh_data['int_depth'] = output['int_depth'].to('cpu')
			else:
				mesh_data=None
			
			image_logger(log_images,sample['name'][0],mesh_data=mesh_data)
			
			
