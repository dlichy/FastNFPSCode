import torch
import torch.nn as nn
from training_utils import loss_functions
from data_processing.depth_to_normals import get_normals_from_depth_list


class Criterion(nn.Module):
	def __init__(self, depth_weight, normal_weight, nfd_weight):
    		super(Criterion,self).__init__()
    		
    		self.key_weight = {'depth_scales': depth_weight, 'normal_scales': normal_weight, 'nfd_scales': nfd_weight}
    		
    		
    
	def list_loss(self,a_s,b_s,masks, base_loss_function):
		
		losses = []
		loss = 0
		for a,b,mask in zip(a_s,b_s,masks):
			loss_i = base_loss_function(a,b,mask)

			loss = loss + loss_i
			losses.append(loss_i)
	
		return loss, losses
		

	#for training output and target should have a list of normal, depth, nfd, target should also have masks
	def forward(self, output, target):
		loss = 0
		loss_dict = {}
		
		target_keys = target.keys()
        	
		output['nfd_scales'] = get_normals_from_depth_list(output['depth_scales'],target['intrinsics'])
        	
		for key in ['normal_scales','depth_scales','nfd_scales']:
			if key in target_keys:
				key_loss, key_loss_scales = self.list_loss(target[key],output[key],target['mask_scales'],loss_functions.L1_loss)		
				loss += self.key_weight[key]*key_loss
				loss_dict[key + '_loss'] = key_loss.item()
				loss_dict[key + '_loss_scale'] = [x.item() for x in key_loss_scales]
			
		with torch.no_grad():
			if 'mean_distance' in target_keys:
				gt_depths_abs = [target['mean_distance'].view(-1,1,1,1)*x for x in target['depth_scales']]
				_, abs_depth_loss_scales = self.list_loss(gt_depths_abs, output['depth_scales'], target['mask_scales'], loss_functions.L1_loss_scale_inv)
				loss_dict['abs_depth_loss_scale'] = [x.item() for x in abs_depth_loss_scales]
				
			if 'normal_scales' in target_keys:
				_, angular_err_scales = self.list_loss(target['normal_scales'],output['normal_scales'], target['mask_scales'], loss_functions.angular_err)
				loss_dict['mae_scale'] = [x.item() for x in angular_err_scales]
         		
			if 'int_depth' in output:
         			gt_depth_abs = target['mean_distance'].view(-1,1,1,1)*target['depth_scales'][-1]
         			abs_depth_loss_int = loss_functions.L1_loss_scale_inv(gt_depth_abs, output['int_depth'].to(gt_depth_abs.device), target['mask_scales'][-1])
         			loss_dict['abs_depth_loss_int'] = abs_depth_loss_int.item()
         			
		if isinstance(loss, torch.Tensor):
			loss_dict['loss'] = loss.item()
		else: 
			loss_dict['loss'] = loss
			       
		return loss, loss_dict
     
