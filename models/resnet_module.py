import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def freeze_bn(m):
	if isinstance(m, nn.BatchNorm2d):
		m.eval()

def forgiving_state_restore(net, loaded_dict):
	net_state_dict = net.state_dict()
	new_loaded_dict = {}
	for k in net_state_dict:
		if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
			new_loaded_dict[k] = loaded_dict[k]
			print(k)
		else:
			logging.info('Skipped loading parameter {}'.format(k))
			print('skip: ' + k)
	net_state_dict.update(new_loaded_dict)
	net.load_state_dict(net_state_dict)
	return net

def conv_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		init.xavier_uniform(m.weight, gain=np.sqrt(2))
		#init.normal(m.weight)
		init.constant(m.bias, 0)

	if classname.find('Linear') != -1:
		init.normal(m.weight)
		init.constant(m.bias,1)

	if classname.find('BatchNorm2d') != -1:
		init.normal(m.weight.data, 1.0, 0.2)
		init.constant(m.bias.data, 0.0)

class conv1x1(nn.Module):
	'''(conv => BN => ReLU)'''
	def __init__(self, in_ch, out_ch):
		super(conv1x1, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 1, stride=1,padding=0),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		x = self.conv(x)
		return x

class conv3x3(nn.Module):
	'''(conv => BN => ReLU)'''
	def __init__(self, in_ch, out_ch):
		super(conv3x3, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3, stride=2,padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		x = self.conv(x)
		return x

# Define a resnet block
class ResnetBlock(nn.Module):
	def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
		super(ResnetBlock, self).__init__()
		self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

	def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
		conv_block = []
		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)

		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
					   norm_layer(dim),
					   nn.ReLU(True)]
		if use_dropout:
			conv_block += [nn.Dropout(0.5)]

		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)
		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
					   norm_layer(dim)]

		return nn.Sequential(*conv_block)

	def forward(self, x):
		out = x + self.conv_block(x)
		return out
