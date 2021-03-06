import torch

def save_checkpoint(save_path, net=None, optimizer=None, epoch=0):
	checkpoint = {}
	checkpoint['epoch'] = epoch
	if net is not None:
		checkpoint['state'] = net.module.state_dict()
	if optimizer is not None:
		checkpoint['optimizer'] = optimizer.state_dict()
	torch.save(checkpoint, save_path)
	
def load_checkpoint(load_path, net=None, optimizer=None, device=None,strict=False):
	checkpoint = torch.load(load_path,map_location = torch.device('cpu'))
	if net is not None:
		net.load_state_dict(checkpoint['state'],strict=strict)
	if optimizer is not None:
		optimizer.load_state_dict(checkpoint['optimizer'])
	return checkpoint['epoch']
