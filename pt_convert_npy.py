import torch
import numpy as np



tensor_data = torch.load('/data/hzb/project/BodyDecoding/data/HCP_phenotype_feature_109_1030.pt')
numpy_array = tensor_data.cpu().numpy()

np.save('/data/hzb/project/BodyDecoding/data/HCP_phenotype_feature_109_1030.npy', numpy_array)