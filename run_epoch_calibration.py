import torch
import matplotlib.pyplot as plt

def prepare_gt(sample,device):
	light_pos = [x[0].to(device) for x in sample['light_data']]
	return light_pos

		
def train_epoch(net, dataloader, sample_preprocessing_fun, device, criterion, optimizer, scalar_logger=None):
	
	net.train()

	for batch_num, sample in enumerate(dataloader):
		
		sample = sample_preprocessing_fun(sample)
		target_light_pos = prepare_gt(sample,device)

		optimizer.zero_grad()
		
		pred_light_pos = net(sample['images'],sample['mask'], sample['intrinsics'])
		
		loss,losses_dict = criterion(pred_light_pos, target_light_pos)
		
		loss.backward()
		optimizer.step()
		
		if scalar_logger is not None:
			scalar_logger(losses_dict,sample['name'][0])
			
			
def eval_epoch(net, dataloader, sample_preprocessing_fun, device, criterion, scalar_logger=None):
	
	net.eval()

	for batch_num, sample in enumerate(dataloader):
		
		sample = sample_preprocessing_fun(sample)
		target_light_pos = prepare_gt(sample,device)


		pred_light_pos = net(sample['images'], sample['mask'], sample['intrinsics'])
		
		loss,losses_dict = criterion(pred_light_pos, target_light_pos)
		
		
		if scalar_logger is not None:
			scalar_logger(losses_dict,sample['name'][0])

