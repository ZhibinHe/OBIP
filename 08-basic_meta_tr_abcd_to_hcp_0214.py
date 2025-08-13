import os
import numpy as np
import argparse
import time
import copy


from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from net.braingnn import Network_regress_score
from sklearn.metrics import classification_report, confusion_matrix
from imports.data_load_hcp import *
from net.configuration_brainlm import BrainLMConfig
import pandas as pd
import pylab
from net.Network_Combine_HCP import *

from torch_geometric.data import Data, Dataset, DataLoader
from net.modeling_brainlm import *

# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def __len__(self):
        return self.A.shape[0]

    def __getitem__(self, idx):
        a = torch.tensor(self.A[idx])
        b = torch.tensor(self.B[idx])
        return Data(x=a, y=b)



torch.manual_seed(123)

EPS = 1e-10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=8, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='/data/hzb/project/BodyDecoding/data/ABCD_pcp/Schaefer/filt_noglobal', help='root directory of the dataset')
parser.add_argument('--fold', type=int, default=0, help='training which fold')
parser.add_argument('--lr', type = float, default=0.00005, help='learning rate')
parser.add_argument('--stepsize', type=int, default=20, help='scheduler step size')
parser.add_argument('--gamma', type=float, default=0.5, help='scheduler shrinking rate')
parser.add_argument('--weightdecay', type=float, default=5e-3, help='regularization')
parser.add_argument('--lamb0', type=float, default=1, help='classification loss weight')
parser.add_argument('--ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--indim', type=int, default=400, help='feature dim')
parser.add_argument('--nroi', type=int, default=400, help='num of ROIs')
parser.add_argument('--nclass', type=int, default=1, help='num of classes')
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--save_model', type=bool, default=True)
parser.add_argument('--optim', type=str, default='Adam', help='optimization method: SGD, Adam')
parser.add_argument('--save_path', type=str, default='./model/', help='path to save model')
parser.add_argument('--phenoroot', type=str, default='/data/hzb/project/BodyDecoding_data/ABCD/ABCD_Phenotype')
parser.add_argument('--stage1_epochs', type=int, default=100, help='number of epochs of training')  #52
parser.add_argument('--num_prediction', type=int, default=115, help='number prediction')
parser.add_argument('--zero_num_prediction', type=int, default=34, help='number prediction')
parser.add_argument('--k_shot', type=int, default=10)
parser.add_argument('--datatootA', type=str, default='/data/hzb/project/BodyDecoding/data/HCP_A_pcp/Schaefer/filt_noglobal', help='root directory of the dataset')
parser.add_argument('--datatootD', type=str, default='/data/hzb/project/BodyDecoding/data/HCP_D_pcp/Schaefer/filt_noglobal', help='root directory of the dataset')
parser.add_argument('--datatootYA', type=str, default='/data/hzb/project/BodyDecoding/data/HCP_pcp/Schaefer/filt_noglobal', help='root directory of the dataset')




# bs=8, lamb0=1,  epoch=8, 0.2043

opt = parser.parse_args()

if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path)

#################### Parameter Initialization #######################
path = opt.dataroot
pathA = opt.datatootA   #######testing dataset
pathD = opt.datatootD   #######testing dataset
pathYA = opt.datatootYA   #######testing dataset



# path2 = opt.datatoot_hcpa   #######testing dataset
# path2 = opt.datatoot_hcpd   #######testing dataset

# config = BrainLMConfig(num_brain_voxels= opt.num_prediction*2)    ##115*2  56 48   num_prediction*2


name = 'ABCD'
save_model = opt.save_model
load_model = opt.load_model
opt_method = opt.optim
num_epoch = opt.n_epochs
stage1_epoch = opt.stage1_epochs
fold = opt.fold
writer = SummaryWriter(os.path.join('./log',str(fold)))








model1 = torch.load('./model/stage1_across_tr_abcd_model_10.pth')

text_feature_ABCD =  torch.load('./model/ABCD_text_115_512_feature.pt')





# train_dataset, val_dataset, test_dataset, text_feature = data_load_hcp_a_fmri(pathA, name, opt)
# train_dataset, val_dataset, test_dataset, text_feature = data_load_hcp_d_fmri(pathD, name, opt)
train_dataset, val_dataset, test_dataset, text_feature = data_load_hcp_fmri(pathYA, name, opt)


train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False)


def test_acc1(loader):
    model1.eval()
    correct = []
    device_cpu = torch.device('cpu')


    with torch.no_grad():

      pred_score = torch.tensor(torch.zeros(1, text_feature_ABCD.shape[0]), device=device)
      label_score = torch.tensor(torch.zeros(1, text_feature.shape[0]), device=device)
      fusion_feature = torch.tensor(torch.zeros(1, text_feature_ABCD.shape[0], text_feature_ABCD.shape[1] * 2),
                                    device=device_cpu)
      # TT=0

      for data in loader:
        data = data.to(device)
        # outputs = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
        score_predict, regress_weight, _ = model1(data.x.view(int(data.x.shape[0]/opt.nroi),opt.nroi, opt.nroi), text_feature_ABCD.to(device))  #
        score_predict1 = torch.stack(score_predict)[:, :, 0].T
        pred_score = torch.cat((pred_score, score_predict1), dim=0)
        label_score = torch.cat((label_score, data.y), dim=0)


    return label_score[1:,:], pred_score[1:,:] #correct,  correct, fusion_feature[1:,:,:],  regress_weight1,  #/ len(loader.dataset)





nan_mask = torch.isnan(train_loader.dataset.data.y)
valid_rows = ~nan_mask.any(dim=1)
valid_row_indices = torch.nonzero(valid_rows, as_tuple=True)[0]
valid_row_indices[0:opt.k_shot]

dataset_x = train_loader.dataset.data.x.view(train_loader.dataset.data.y.shape[0],400,400)
dataset_y = train_loader.dataset.data.y

k_shot_dataset_x = dataset_x[valid_row_indices[0:opt.k_shot],:,:]
k_shot_dataset_y = dataset_y[valid_row_indices[0:opt.k_shot],:]


score_predict, _, _ = model1(k_shot_dataset_x.to(device),text_feature_ABCD.to(device))
score_predict = torch.cat(score_predict, dim=1)

data1 = k_shot_dataset_y.to(device)  # 形状为 (5, 34)
data2 = score_predict  # 形状为 (100, 34)

most_similar_indices = []

for i in range(data1.size(1)):
    col1 = data1[:, i]
    similarities = torch.nn.functional.cosine_similarity(col1.unsqueeze(0), data2.transpose(0, 1))

    most_similar_index = torch.argmax(similarities).item()
    most_similar_indices.append(most_similar_index)

most_similar_indices_tensor = torch.tensor(most_similar_indices)


label_score_tr, pred_score_tr = test_acc1(train_loader)


pred_score_tr = pred_score_tr[:,most_similar_indices_tensor]


filename111 = f'./log/meta_basic/HCP_YA/K_{opt.k_shot}/label_sys_corr_epoch_0.npy'
np.save(filename111, pred_score_tr.cpu().numpy())


filename111 = f'./log/meta_basic/HCP_YA/K_{opt.k_shot}/label_score_epoch_0.npy'
np.save(filename111, label_score_tr.cpu().numpy())





print('train_end')




