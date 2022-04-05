from plyfile import PlyData, PlyElement
import numpy as np


#vert_props is list of length len(prop_names) all numpy arrays of length N. faces is Mx3
def write_ply(save_name,vert_props,prop_names=['x','y','z','nx','ny','nz'],prop_types=None,faces=None,as_text=False):
	
	#make vertex element
	if prop_types is None:
		prop_types = ['float32' for _ in prop_names]
	
	vert_props_list = [x.tolist() for x in vert_props]
	
	struct_array_data = [t for t in zip(*vert_props_list)]
	
	data_dtype = [x for x in zip(prop_names,prop_types)]
	struct_array = np.array(struct_array_data,dtype=data_dtype)
	
	vert_elem = PlyElement.describe(struct_array,'vertex')
	
	all_elem = [vert_elem]
	
	#make face element
	if faces is not None:
		ply_faces = np.empty(len(faces), dtype=[('vertex_indices', 'i4', (3,))])
		ply_faces['vertex_indices'] = faces
		face_elem = PlyElement.describe(ply_faces, 'face')
		all_elem.append(face_elem)

	ply_data =PlyData( all_elem,text=as_text )
	ply_data.write(save_name)


def read_ply(file_name,prop_names=['x','y','z','nx','ny','nz']):
	plydata = PlyData.read(file_name)
	elem_names = [e.name for e in plydata.elements]
	vert_props = []
	for prop_name in prop_names:
		vert_prop = plydata['vertex'][prop_name]
		vert_props.append(vert_prop)
	if 'face' in elem_names:
		faces = plydata['face'].data
		faces = np.concatenate(faces.tolist(),axis=0)
	else:
		faces = None
		
	return vert_props, faces
	
