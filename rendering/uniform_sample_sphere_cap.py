import numpy as np

def uniform_sample_sphere_cap(max_phi,num_samples):
	theta = 2*np.pi*np.random.rand(num_samples)
	
	#phi in [0,max_phi]
	cos_min_phi = np.cos(max_phi)
	cos_phi = (1-cos_min_phi)*np.random.rand(num_samples) + cos_min_phi
	
	phi = np.arccos(cos_phi)
	
	x = np.sin(phi)*np.cos(theta)
	y = np.sin(phi)*np.sin(theta)
	z = np.cos(phi)
	return np.stack((x,y,z))
	

