import os
import numpy as np
import argparse
import time
import copy

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

from imports.ABIDEDataset import HCPfmriScoreDataset, HCPfmriScoreDataset_sbjnum
from torch_geometric.data import DataLoader
from net.braingnn import Network_regress_score
from imports.utils import train_val_test_split_hcp
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from imports.vit import ViT
from net.Network_Dual_ViT import *
from net.Network_Pred import *
from net.Network_Combine import *
from net.models_mae import *
from imports.data_load_hcp import *
from net.configuration_brainlm import BrainLMConfig
from net.Network_Combine_HCP import *

import pandas as pd
import pylab

# 333 node Epoch: 018 0.1822354,  0m 7s
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


config = BrainLMConfig()

torch.manual_seed(123)

EPS = 1e-10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=16, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='/data/hzb/project/BodyDecoding/data/ABCD_pcp/Schaefer/filt_noglobal', help='root directory of the dataset')
parser.add_argument('--fold', type=int, default=0, help='training which fold')
parser.add_argument('--lr', type = float, default=0.00005, help='learning rate')
parser.add_argument('--stepsize', type=int, default=20, help='scheduler step size')
parser.add_argument('--gamma', type=float, default=0.5, help='scheduler shrinking rate')
parser.add_argument('--weightdecay', type=float, default=5e-3, help='regularization')
parser.add_argument('--lamb0', type=float, default=1, help='classification loss weight')
parser.add_argument('--lamb1', type=float, default=0, help='s1 unit regularization')
parser.add_argument('--lamb2', type=float, default=0, help='s2 unit regularization')
parser.add_argument('--lamb3', type=float, default=0, help='s1 entropy regularization')
parser.add_argument('--lamb4', type=float, default=0, help='s2 entropy regularization')
parser.add_argument('--lamb5', type=float, default=0, help='s1 consistence regularization')
parser.add_argument('--layer', type=int, default=2, help='number of GNN layers')
parser.add_argument('--ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--indim', type=int, default=400, help='feature dim')
parser.add_argument('--nroi', type=int, default=400, help='num of ROIs')
parser.add_argument('--nclass', type=int, default=1, help='num of classes')
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--save_model', type=bool, default=True)
parser.add_argument('--optim', type=str, default='Adam', help='optimization method: SGD, Adam')
parser.add_argument('--save_path', type=str, default='./model/', help='path to save model')
parser.add_argument('--csvroot', type=str, default='/data/hzb/project/BodyDecoding/data/ABCD_pcp/ABCD_Phenotype')
parser.add_argument('--datadir', type=str, default='/data/hzb/project/BodyDecoding/data')

parser.add_argument('--csv_head_num', type=int, default=3)
parser.add_argument('--out_txt_name', type=str, default='output.txt')
parser.add_argument('--stage1_epochs', type=int, default=16, help='number of epochs of training')  #52
parser.add_argument('--datatoot_hcp', type=str, default='/data/hzb/project/BodyDecoding/data/ABCD_pcp/Schaefer/filt_noglobal/', help='root directory of the dataset')

# bs=8, lamb0=1,  epoch=8, 0.2043

opt = parser.parse_args()

if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path)

#################### Parameter Initialization #######################
path = opt.dataroot
path2 = opt.datatoot_hcp
name = 'ABCD'
save_model = opt.save_model
load_model = opt.load_model
opt_method = opt.optim
num_epoch = opt.n_epochs
stage1_epoch = opt.stage1_epochs
fold = opt.fold
writer = SummaryWriter(os.path.join('./log',str(fold)))

################## Define Dataloader ##################################
train_dataset, val_dataset, test_dataset, text_feature = data_load_abcd_fmri(path, name, opt)


# train_dataset, val_dataset, test_dataset, text_feature, dataset, train_index1, val_index1 = data_load_hcpa_t1fmri(path, path2, name, opt)
text_feature_all = text_feature
# text_feature = text_feature[:, :24, :]
text_feature_zero = text_feature_all.repeat(opt.batchSize, 1, 1)


train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)



model1 = torch.load('./model/stage1_fmri_best_ABCD_1118.pth')


# model1 = CombinedModel_sink_feature400().to(device)
# model2 =BrainLMDecoder_mask(config, num_patches=196).to(device)



optimizer1 = torch.optim.Adam(model1.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)
# optimizer2 = torch.optim.Adam(model2.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)



scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=opt.stepsize, gamma=opt.gamma)
# scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=opt.stepsize, gamma=opt.gamma)

############################### Define Other Loss Functions ########################################

loss_fn = nn.MSELoss(reduction='none')
loss_rec = nn.L1Loss(reduction='none')


###################### Network Training Function#####################################
def train1(epoch):
    scheduler1.step()
    model1.train()
    loss_all = 0
    step = 0
    for data in train_loader:
        data = data.to(device)
        optimizer1.zero_grad()


        score_predict , _, _= model1(data.x.view(int(data.x.shape[0]/opt.nroi),opt.nroi, opt.nroi), text_feature) #
        data.y = torch.tensor(data.y, dtype=torch.float)
        data.y[torch.where(torch.isnan(data.y) == True)] = 0
        nan_mask = torch.isnan(data.y)
        isnan_matrix = torch.zeros_like(data.y)
        isnan_matrix[~nan_mask] = 1
        column_losses = loss_fn(torch.stack(score_predict)[:,:,0].T *isnan_matrix, data.y * isnan_matrix)
        # 按列求和或取平均
        loss_c = torch.sum(column_losses)
        loss = opt.lamb0*loss_c #+ opt.lamb1 * correlation_loss #+ opt.lamb2 * loss_p2 \
        step = step + 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model1.parameters(), max_norm=4.0)
        loss_all += loss.item() * data.num_graphs
        optimizer1.step()
        print(step)


    return loss_all / len(train_dataset) #, s1_arr, s2_arr ,w1,w2





def test_acc1(loader):
    model1.eval()
    correct = []
    device_cpu = torch.device('cpu')


    with torch.no_grad():
      pred_score = torch.tensor(torch.zeros(1, text_feature.shape[0]), device=device)
      label_score = torch.tensor(torch.zeros(1, text_feature.shape[0]), device=device)
      fusion_feature = torch.tensor(torch.zeros(1, text_feature.shape[0], text_feature.shape[1] * 2),
                                    device=device_cpu)
      TT=0
      for data in loader:
        data = data.to(device)
        # outputs = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
        score_predict, regress_weight, feature_cat = model1(data.x.view(int(data.x.shape[0]/opt.nroi),opt.nroi, opt.nroi), text_feature)  #
        score_predict1 = torch.stack(score_predict)[:, :, 0].T
        pred_score = torch.cat((pred_score, score_predict1), dim=0)
        label_score = torch.cat((label_score, data.y), dim=0)
        fusion_feature = torch.cat((fusion_feature, feature_cat.cpu()), dim=0)
        TT = TT+1
        print(TT)

    label_score[torch.where(torch.isnan(label_score) == True)] = 0
    for i in range(pred_score.shape[1]):
        correct_task = np.corrcoef(pred_score[1:, i].detach().cpu().numpy().T, label_score[1:,i].detach().cpu().numpy().T)[0, 1]
        correct.append(correct_task)




    correct = np.sum(correct)/pred_score.shape[1]
    regress_weight = torch.squeeze(torch.stack(regress_weight), axis=1)
    regress_weight1 = regress_weight.cpu()
    del regress_weight

    return correct,  correct, fusion_feature[1:,:,:],  regress_weight1, label_score[1:,:], pred_score[1:,:] #/ len(loader.dataset)













def test_loss1(loader,epoch):
    # print('testing...........')
    model1.eval()
    loss_all = 0
    loss_c = []
    for data in loader:
        data = data.to(device)


        score_predict , _, _= model1(data.x.view(int(data.x.shape[0]/opt.nroi),opt.nroi, opt.nroi), text_feature) #
        data.y = torch.tensor(data.y, dtype=torch.float)
        data.y[torch.where(torch.isnan(data.y) == True)] = 0
        nan_mask = torch.isnan(data.y)
        isnan_matrix = torch.zeros_like(data.y)
        isnan_matrix[~nan_mask] = 1
        column_losses = loss_fn(torch.stack(score_predict)[:,:,0].T *isnan_matrix, data.y * isnan_matrix)
        # 按列求和或取平均
        loss_c = torch.sum(column_losses)
        loss = opt.lamb0*loss_c #+ opt.lamb1 * correlation_loss #+ opt.lamb2 * loss_p2 \
        loss_all += loss.item() * data.num_graphs
    return loss_all / len(loader.dataset)

############################   Model Training #########################################
best_model_wts = copy.deepcopy(model1.state_dict())
best_loss = 1e10
val_acc_list = []


for epoch in range(0,stage1_epoch):
    since  = time.time()
    # tr_loss, s1_arr, s2_arr, w1, w2 = train(epoch)
    val_acc, val_rmse, fusion_feature_val, regress_weight_val, label_score_val, pred_score_val = test_acc1(train_loader)
    # tr_loss= train1(epoch)
    if epoch==14:
        tr_acc, tr_rmse, fusion_feature_tr, regress_weight_tr, label_score_tr, pred_score_tr = test_acc1(train_loader)
        val_acc, val_rmse, fusion_feature_val, regress_weight_val, label_score_val, pred_score_val = test_acc1(val_loader)
        torch.save(model1, './model/stage1_fmri_best_ABCD_1118.pth')
        torch.save([train_loader, val_loader,text_feature, text_feature_zero, fusion_feature_tr, fusion_feature_val, label_score_tr, label_score_val, regress_weight_tr, regress_weight_val], './model/stage1_dataset_ABCD_fmri_1118.pt')

    # val_acc_list.append(val_acc)
    # val_loss = test_loss1(val_loader,epoch)
    time_elapsed = time.time() - since
    time_elapsed = time.time() - since
    print('*====**')
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Epoch: {:03d}, Train Loss: {:.7f}, '
    #       'Train Acc: {:.7f},Train rmse: {:.7f}, Test Loss: {:.7f}, Test Acc: {:.7f}, Test rmse: {:.7f}, '.format(epoch, tr_loss,
    #                                                    tr_acc, tr_rmse, val_loss, val_acc, val_rmse))

# torch.save(model1, './model/stage1_fmri_best_ABCD_1118.pth')

# torch.save([train_loader, val_loader,text_feature, text_feature_zero, fusion_feature_tr, fusion_feature_val, label_score_tr, label_score_val, regress_weight_tr, regress_weight_val], './model/stage1_dataset_ABCD_fmri_1118.pt')

#########################################################
fusion_feature = fusion_feature_tr
fusion_feature = fusion_feature.view(fusion_feature.shape[0], fusion_feature.shape[1]*fusion_feature.shape[2])
train_dataset_stage2 = CustomDataset(fusion_feature, label_score_tr)
train_loader_stage2 = DataLoader(train_dataset_stage2, batch_size=opt.batchSize, shuffle=False, drop_last=True)
# torch.matmul(fusion_feature_tr[0,0,:].T, regress_weight_tr[0,:])
fusion_feature = fusion_feature_val
fusion_feature = fusion_feature.view(fusion_feature.shape[0], fusion_feature.shape[1]*fusion_feature.shape[2])
val_dataset_stage2 = CustomDataset(fusion_feature, label_score_val)
val_dataset_stage2 = DataLoader(val_dataset_stage2, batch_size=opt.batchSize, shuffle=True, drop_last=True)


regress_weight_tr_repeat = regress_weight_tr.unsqueeze(0).repeat(opt.batchSize, 1, 1).to(device)

# torch.save([train_loader_stage2, val_dataset_stage2, regress_weight_tr_repeat], './model/stage1_dataset.pt')

#####################################################################


for epoch in range(stage1_epoch, 1000):
    since  = time.time()

    tr_loss=  train2(epoch, train_loader_stage2)
    # label_train1_corr_mean,  label_sys_corr_mean, train1_sys_corr_mean= test_acc2(train_loader_stage2)    #   test_acc2(val_dataset_stage2)
    # label_sys_corr= test_acc_zero(train_loader_stage2)    #   test_acc2(val_dataset_stage2)

    # tr_acc, tr_rmse, fusion_feature_tr,  regress_weight_tr= test_acc1(train_loader)
    # val_acc, val_rmse, fusion_feature_val,  regress_weight_val = test_acc1(val_loader)
    #
    # val_acc_list.append(val_acc)
    # val_loss = test_loss1(val_loader,epoch)


    time_elapsed = time.time() - since


    print('*====**')
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Epoch: {:03d}, Train Loss: {:.7f}, '.format(epoch, tr_loss))



    # print('*====**')
    # print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Epoch: {:03d}, Train Loss: {:.7f}, label_train1_corr_mean: {:.7f}, label_sys_corr_mean: {:.7f}, train1_sys_corr_mean{:.7f}'.format(epoch, tr_loss, label_train1_corr_mean,  label_sys_corr_mean, train1_sys_corr_mean))


    # print('Epoch: {:03d}, Train Loss: {:.7f}, '
    #       'Train Acc: {:.7f},Train rmse: {:.7f}, Test Loss: {:.7f}, Test Acc: {:.7f}, Test rmse: {:.7f}, '.format(epoch, tr_loss,
    #                                                    tr_acc, tr_rmse, val_loss, val_acc, val_rmse))

    writer.add_scalars('Acc',{'train_acc':tr_acc,'val_acc':val_acc},  epoch)
    writer.add_scalars('Loss', {'train_loss': tr_loss, 'val_loss': val_loss},  epoch)










######################### Testing on testing set ######################################
#######################################################################################

if opt.load_model1:
    model1 = Network_regress_score(opt.indim,opt.ratio,opt.nclass).to(device)
    model1.load_state_dict(torch.load(os.path.join(opt.save_path,str(fold)+'.pth')))
    model1.eval()
    preds = []
    correct = 0
    for data in val_loader:
        data = data.to(device)
        outputs= model1(data.x, data.edge_index, data.batch, data.edge_attr,data.pos)
        pred = outputs[0].max(1)[1]
        preds.append(pred.cpu().detach().numpy())
        correct += pred.eq(data.y).sum().item()
    preds = np.concatenate(preds,axis=0)
    trues = val_dataset.data.y.cpu().detach().numpy()
    cm = confusion_matrix(trues,preds)
    print("Confusion matrix")
    print(classification_report(trues, preds))

else:
   model1.load_state_dict(best_model_wts)
   model1.eval()
   test_accuracy , test_rmse= test_acc1(test_loader)
   test_l= test_loss1(test_loader,0)
   print("===========================")
   print("Test Acc: {:.7f}, Test Loss: {:.7f} ".format(test_accuracy, test_l))
   print(opt)

