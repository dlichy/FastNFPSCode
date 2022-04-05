import data_processing.optical_flow_funs as OF
import torch

	
def depth_along_ray_to_depth_z(depth, n_intrinsics):
	#depth (b,1,m,n)
	#intrinisics (b,3,3)
	#return (b,1,m,n)
	intrinsics = OF.normalized_intrinsics_to_pixel_intrinsics(n_intrinsics,depth.shape[2:4])
	grid = OF.get_coordinate_grid(depth.shape[2:4],device=depth.device).repeat(depth.size(0),1,1,1)
	dirs = OF.pixel_coords_to_directions(grid,intrinsics)
	pc = depth.squeeze(1).unsqueeze(3)*torch.nn.functional.normalize(dirs,dim=3)
	depth_z = pc[:,:,:,2:3].squeeze(3).unsqueeze(1)
	return depth_z
