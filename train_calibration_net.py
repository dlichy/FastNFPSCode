import torch.optim as optim
from models.calibration_criterion import *
from models.calibration_net import *
from datasets.synthetic_sculpture_dataset import *
from datasets.luces_dataset import *
from models.save_load_checkpoint import *
from data_processing.common_transforms import *
from datasets.synthetic_batch_augmentations import *
from run_epoch_calibration import train_epoch, eval_epoch
from run_epoch import PreprocessSynthetic, PreprocessReal
from training_utils.logger import *
from rendering.render_like_luces_more_general import *
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('logdir', default=None, help='the path to store logging information and models and models')
parser.add_argument('--gpu', default=False,  action='store_true', help='enable to run on gpu')
# The location of training set
parser.add_argument('--syn_dataset_root', type=str, help='path to random object dataset', default='')
parser.add_argument('--luces_dataset_root', type=str, help='path to luces dataset', default='')
parser.add_argument('--merl_path', type=str, help='path to MERL BRDF dataset', default='')
# The basic training setting
parser.add_argument('--epochs', type=int, default=20, help='the number of epochs for training')
parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
# The training weight 
parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')
#logging options
parser.add_argument('--scalars_to_log', type=list, default=['loss'], help='the scalars to log')
parser.add_argument('--test_scalar_lf', type=int, default=20, help='frequency to log scalars during testing')
parser.add_argument('--train_scalar_lf', type=int, default=50, help='frequency to log scalars during training')
parser.add_argument('--checkpoint_freq', type=int, default=2, help='how frequently to save model weights')
#resume training from checkpoint
parser.add_argument('--checkpoint', default='None', help='path to checkpoint to load')

#rendering args
parser.add_argument('--num_train_lights',type=int,default=10)
parser.add_argument('--admissible_region_radius',type=float, default=0.75)
parser.add_argument('--admissible_region_extent',type=tuple, default=(-0.15,0.15))
parser.add_argument('--admissible_region_dir',type=float, default=np.pi/6)

opt = parser.parse_args()
if opt.gpu:
	device = 'cuda'
else:
	if torch.cuda.is_available():
		import warnings
		warnings.warn('running on CPU but GPUs detected. Add arg \"--gpu\" to run on gpu')
	device='cpu'




train_data = SyntheticSculptureDataset(opt.syn_dataset_root, mode='train',domains_to_load = ['albedo','normal','depth','mask','roughness'], transform=syn_training_transforms_low_res)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=8)

syn_test_data = SyntheticSculptureDataset(opt.syn_dataset_root, mode='test',domains_to_load = ['albedo','normal','depth','mask','roughness'], transform=syn_test_transforms_low_res)
syn_test_loader = torch.utils.data.DataLoader(syn_test_data, batch_size=opt.batch_size, shuffle=False, num_workers=8)


if opt.luces_dataset_root:
	luces_test_data = LucesDataset(opt.luces_dataset_root, raw=False, transform=luces_transforms_low_res)
	luces_test_loader = torch.utils.data.DataLoader(luces_test_data, batch_size=1, shuffle=False, num_workers=1)
	luces_preprocessing_fun = PreprocessReal(device)


#setup synthetic rendering
light_arg_generator = GeneratedLightsRandom(opt.num_train_lights,pos_radius=opt.admissible_region_radius, pos_z_offset_range=opt.admissible_region_extent, cap_max_phi=opt.admissible_region_dir)
renderer = DataGenLikeLuces(light_arg_generator, 0.5, opt.merl_path,device)
synthetic_preprocessing_fun = PreprocessSynthetic(renderer, device)


#setup network
net = CalibrationNet(batch_norm=True)
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=opt.lr)

if opt.checkpoint == 'None':
	start_epoch = 0
else:
	start_epoch = load_checkpoint(opt.checkpoint, net=net, optimizer=optimizer)

if opt.gpu:
	net = nn.DataParallel(net)	

criterion = CalibrationCriterion()

#make logdir
if not os.path.exists(opt.logdir):
	os.mkdir(opt.logdir)


for epoch in range(start_epoch,opt.epochs):
	epoch_dir = os.path.join(opt.logdir,'epoch_{}'.format(epoch))
	if not os.path.exists(epoch_dir):
		os.mkdir(epoch_dir)
	
	#train
	scalar_logger = ScalarLogger(os.path.join(epoch_dir,'train_log.txt'), log_freq=opt.train_scalar_lf, keys=opt.scalars_to_log)
	train_epoch(net, train_loader,synthetic_preprocessing_fun, device,  criterion, optimizer, scalar_logger=scalar_logger)
	scalar_logger.summarize()
	
	
	#test synthetic
	scalar_logger = ScalarLogger(os.path.join(epoch_dir,'eval_log.txt'), log_freq=opt.test_scalar_lf, keys=opt.scalars_to_log)
	with torch.no_grad():
		eval_epoch(net, syn_test_loader, synthetic_preprocessing_fun, device,  criterion=criterion, scalar_logger=scalar_logger)	
	scalar_logger.summarize()
	
	#test luces
	if opt.luces_dataset_root:
		scalar_logger = ScalarLogger(os.path.join(epoch_dir,'eval_log_luces.txt'), log_freq=1, keys=opt.scalars_to_log)
		with torch.no_grad():
			eval_epoch(net, luces_test_loader, luces_preprocessing_fun, device,  criterion=criterion, scalar_logger=scalar_logger)
		scalar_logger.summarize()
	
	#checkpoint
	if epoch % opt.checkpoint_freq == 0:
		save_checkpoint(os.path.join(epoch_dir,'checkpoint_{}.pth'.format(epoch)), net=net, optimizer=optimizer)
	
	
	
