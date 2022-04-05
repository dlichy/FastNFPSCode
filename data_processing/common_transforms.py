from data_processing.sample_transforms import *
from datasets.synthetic_dataset_utils import *
from datasets.luces_dataset import ScaleByPhi, NormalizeByMeanDepthLuces
from data_processing.pad_to_power2 import *
from data_processing.crop_based_on_mask import *

#synthetic datasets
syn_training_transforms = transforms.Compose([
      ErodeMask(keys=['mask'],mask_erosion_size=(6,6)),
      RandomCropResize((512,512),keys=['albedo','mask','depth','normal','roughness']),
      NormalizeByMeanDepthSynthetic(),
      MyToTensor(keys=['albedo','mask','depth','normal','roughness'])
      ])
      
syn_test_transforms = transforms.Compose([
      ErodeMask(keys=['mask'],mask_erosion_size=(6,6)),
      NormalizeByMeanDepthSynthetic(),
      MyToTensor(keys=['albedo','mask','depth','normal','roughness'])
      ])
      

#luces data loading
luces_transforms = transforms.Compose([
      #Resize((682,512),keys=['normal','depth','mask'],list_keys=['images']),
       ScaleByPhi(),
       NormalizeByMeanDepthLuces(),
       PadSquareToPower2Intrinsics(keys=['normal','depth','mask'],list_keys=['images']),
       MyToTensor(keys=['normal','depth','mask'],list_keys=['images'])
     ])
     
     
#low res transforms for training calibration network
syn_training_transforms_low_res  = transforms.Compose([
      RandomCropResize((128,128),keys=['albedo','mask','depth','normal','roughness']),
      ErodeMask(keys=['mask'],mask_erosion_size=(3,3)),
      NormalizeByMeanDepthSynthetic(),
      MyToTensor(keys=['albedo','mask','depth','normal','roughness'])
      ])
     

syn_test_transforms_low_res = transforms.Compose([
      Resize((128,128),keys=['albedo','mask','depth','normal','roughness']),
      ErodeMask(keys=['mask'],mask_erosion_size=(3,3)),
      NormalizeByMeanDepthSynthetic(),
      MyToTensor(keys=['albedo','mask','depth','normal','roughness'])
      ])
      

luces_transforms_low_res = transforms.Compose([
       Resize((171,128),keys=['normal','depth','mask'],list_keys=['images']),
       ScaleByPhi(),
       NormalizeByMeanDepthLuces(),
       PadSquareToPower2Intrinsics(keys=['normal','depth','mask'],list_keys=['images']),
       MyToTensor(keys=['normal','depth','mask'],list_keys=['images'])
     ])
     
     
stardard_transforms = transforms.Compose([
	CropBasedOnMask('mask',20,square=True, keys=['mask'], list_keys=['images']),
	Resize((512,512),keys=['mask'], list_keys=['images']),
	MyToTensor(keys=['mask'],list_keys=['images'])
      ])


