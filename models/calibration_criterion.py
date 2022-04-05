import torch
import torch.nn as nn


class CalibrationCriterion(nn.Module):
	def __init__(self):
    		super(CalibrationCriterion,self).__init__()
    		
    		
	#list of (b,3)
	def forward(self, pred_light_pos, gt_light_pos):
		loss_dict = {}	
		loss = 0
	
		N = len(pred_light_pos)
		for i in range(N):
			loss += torch.mean(torch.sum((gt_light_pos[i]-pred_light_pos[i])**2,dim=1))
		
		loss = loss/N
		loss_dict['loss'] = loss.item()
		return loss, loss_dict
     
