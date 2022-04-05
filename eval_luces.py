import torch.optim as optim
from models.criterion import *
from models.multiscale_net import *
from models.calibration_net import *
from datasets.luces_dataset import *
from models.save_load_checkpoint import *
from data_processing.common_transforms import *
from run_epoch import  eval_epoch, PreprocessReal
from training_utils.logger import *
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('logdir', default=None, help='the path to store logging information and models and models')
parser.add_argument('--gpu', default=False,  action='store_true', help='enable to run on gpu')

parser.add_argument('--luces_dataset_root', type=str, help='path to luces dataset', default='')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
# The training weight 
parser.add_argument('--depth_weight', type=float, default=1.0, help='the weight for the diffuse component')
parser.add_argument('--normal_weight', type=float, default=1.0, help='the weight for the diffuse component')
parser.add_argument('--nfd_weight', type=float, default=1.0, help='the weight for the roughness component')
#logging options
parser.add_argument('--scalars_to_log', type=list, default=['loss','normal_scales_loss','depth_scales_loss','nfd_scales_loss','normal_scales_loss_scale','depth_scales_loss_scale','nfd_scales_loss_scale','abs_depth_loss_scale','mae_scale','abs_depth_loss_int'], help='the scalars to log')
parser.add_argument('--image_logger_keys', type=list, default=['target_depth_scales','target_normal_scales','output_depth_scales','output_normal_scales','output_int_depth'])
parser.add_argument('--test_scalar_lf', type=int, default=1, help='frequency to log scalars during testing')
parser.add_argument('--test_image_lf', type=int, default=1, help='frequency to log images during testing')
parser.add_argument('--save_first_batch_only', default=True,  action='store_true', help='only save outputs at largest scale')
#checkpoints tp load
parser.add_argument('--checkpoint', default='None', help='path to checkpoint to load')
parser.add_argument('--uncalibrated',default=False, action='store_true', help='use calibration network')
parser.add_argument('--calib_net_checkpoint', type=str)
#mesh logging
parser.add_argument('--post_integrate_normals', default=True, action='store_false')
parser.add_argument('--log_meshes', default=True, action='store_false')


opt = parser.parse_args()
if opt.gpu:
	device = 'cuda'
else:
	if torch.cuda.is_available():
		import warnings
		warnings.warn('running on CPU but GPUs detected. Add arg \"--gpu\" to run on gpu')
	device='cpu'



luces_test_data = LucesDataset(opt.luces_dataset_root, raw=False, transform=luces_transforms)
luces_test_loader = torch.utils.data.DataLoader(luces_test_data, batch_size=1, shuffle=False, num_workers=1)
luces_preprocessing_fun = PreprocessReal(device)


#setup network
net=MultiscaleNet()
net.to(device)
load_checkpoint(opt.checkpoint, net=net)


if opt.uncalibrated:
	calib_net = CalibrationNet(batch_norm=True)
	calib_net.to(device)
	load_checkpoint(opt.calib_net_checkpoint, net=calib_net)
else:
	calib_net = None

if opt.gpu:
	net = nn.DataParallel(net)

criterion = Criterion(opt.depth_weight, opt.normal_weight, opt.nfd_weight)

#make logdir
if not os.path.exists(opt.logdir):
	os.mkdir(opt.logdir)
test_image_dir = os.path.join(opt.logdir,'images')
if not os.path.exists(test_image_dir):
	os.mkdir(test_image_dir)

scalar_logger = ScalarLogger(os.path.join(opt.logdir,'eval_log_luces.txt'), log_freq=1, keys=opt.scalars_to_log)
image_logger = ImageLogger(test_image_dir,log_freq=1,save_first_batch_only=opt.save_first_batch_only,keys=opt.image_logger_keys)
with torch.no_grad():
	eval_epoch(net, luces_test_loader, luces_preprocessing_fun, device,  criterion=criterion, scalar_logger=scalar_logger, image_logger=image_logger,calibration_net=calib_net, post_integrate_normals=opt.post_integrate_normals, log_meshes=opt.log_meshes)
scalar_logger.summarize()
	
	
	
	
	
