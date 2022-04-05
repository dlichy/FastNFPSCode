import data_processing.optical_flow_funs as OF
import torch

#intrinsics should be normalized
def resample_by_intrinsics(src_image, src_intrinsics, trg_intrinsics, trg_size):
	grid = OF.get_coordinate_grid(trg_size,device=src_image.device).repeat(src_image.size(0),1,1,1)
	grid_n = OF.pixel_coords_to_normalized_coords(grid,trg_size)
	T = torch.bmm(src_intrinsics,torch.inverse(trg_intrinsics))
	pts = OF.apply_affine(T,grid_n)
	trg_image = torch.nn.functional.grid_sample(src_image,pts, align_corners=False)
	return trg_image
	

