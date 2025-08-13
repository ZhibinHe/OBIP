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
from net.models_mae import *
from imports.data_load_hcp import *
from net.configuration_brainlm import BrainLMConfig

import pandas as pd
import pylab
from net.Network_Combine_HCP import *

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


config = BrainLMConfig(num_brain_voxels= 630)

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
parser.add_argument('--datatoot_hcp', type=str, default='/data/hzb/project/BodyDecoding/data/HCP_pcp/Schaefer/filt_noglobal', help='root directory of the dataset')

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
def test_acc1(loader):
    model1.eval()
    correct = []
    device_cpu = torch.device('cpu')

    with torch.no_grad():


      # pred_score = torch.tensor(torch.zeros(1, text_feature_hcp.shape[0]), device=device)
      label_score = torch.tensor(torch.zeros(1, text_feature_hcp.shape[0]), device=device)
      fusion_feature = torch.tensor(torch.zeros(1, text_feature_hcp.shape[0], text_feature_hcp.shape[1]*2), device=device_cpu)

      for data in loader:
        data = data.to(device)
        all_datay = data.y
        feature_cat_tmp = torch.tensor(torch.zeros(opt.batchSize, text_feature_hcp.shape[0], text_feature_hcp.shape[1]*2), device=device)
        text_feature_tmp = text_feature

        for i in range(text_feature_hcp.shape[0]):
            text_feature_tmp[0:1, :] = text_feature_hcp[i:i+1, :]
            score_predict, regress_weight, feature_cat = model1(
                data.x.view(int(data.x.shape[0] / opt.nroi), opt.nroi, opt.nroi), text_feature_tmp)
            feature_cat_tmp[:, i, :] = feature_cat[:, 0, :]


        fusion_feature = torch.cat((fusion_feature, feature_cat_tmp.cpu()), dim=0)
        label_score = torch.cat((label_score, all_datay), dim=0)






    #
    #     text_feature_expand = torch.zeros(315,512).to('cuda:0')
    #     text_feature_expand[:109,:] = text_feature_hcp
    #
    #     score_predict , regress_weight, feature_cat = model1(data.x.view(int(data.x.shape[0]/opt.nroi),opt.nroi, opt.nroi), text_feature_expand)  #
    #
    #     # score_predict1 = torch.stack(score_predict)[:, :, 0].T
    #     # pred_score = torch.cat((pred_score, score_predict1), dim=0)
    #     label_score = torch.cat((label_score, all_datay), dim=0)
    #     fusion_feature = torch.cat((fusion_feature, feature_cat.cpu()), dim=0)
    #
    # for i in range(pred_score.shape[1]):
    #     correct_task = np.corrcoef(pred_score[1:, i].detach().cpu().numpy().T, label_score[1:,i].detach().cpu().numpy().T)[0, 1]
    #     correct.append(correct_task)
    #
    # # if pred_score.shape[0] != 2586:
    # #     print(correct[0])
    # #     np.save('vit_aug_sink_lossop_' +str(epoch)+ '.npy', np.array(correct))
    #
    #
    # correct = np.sum(correct)/pred_score.shape[1]
    # regress_weight = torch.squeeze(torch.stack(regress_weight), axis=1)
    # regress_weight1 = regress_weight.cpu()
    # del regress_weight

    return fusion_feature[1:,:,:] , label_score[1:,:]#,  regress_weight1, , pred_score[1:,:] #/ len(loader.dataset)

#################################

model1 = torch.load('./model/stage1_fmri_best_ABCD_1118.pth')
train_dataset, val_dataset, test_dataset, text_feature = data_load_abcd_fmri(path, name, opt)

# train_dataset_hcp, val_dataset_hcp, test_dataset_hcp, text_feature_hcp = data_load_hcp_fmri(path2, 'HCP', opt)

# train_loader_hcp = DataLoader(train_dataset_hcp, batch_size=opt.batchSize, shuffle=True)
# hcp_feature, hcp_label = test_acc1(train_loader_hcp)

train_loader, val_loader,text_feature, text_feature_zero, fusion_feature_tr, fusion_feature_val, label_score_tr, label_score_val, regress_weight_tr, regress_weight_val = torch.load('./model/stage1_dataset_ABCD_fmri_1118.pt')
hcp_feature, hcp_label = torch.load('./model/stage3_HCP_fmri_feature_label_1119.pt')

text_feature_all = text_feature
# text_feature = text_feature[:, :24, :]
text_feature_zero = hcp_feature[0,:,512:].repeat(opt.batchSize, 1, 1)

# torch.save([hcp_feature[:860,:], hcp_label[:860,:]], './model/stage3_HCP_fmri_feature_label_1119.pt')


#
# train_dataset, val_dataset, test_dataset, text_feature, dataset, train_index1, val_index1 = data_load_hcp_t1fmri(path, path2, name, opt)
# text_feature_all = text_feature
# text_feature = text_feature[:, :28, :]
# text_feature_zero = text_feature_all[1,28:,:].repeat(8, 1, 1)
#
#
# train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)
#
#
# model1 = CombinedModel_sink_feature().to(device)
# model2 = MaskedAutoencoderViT_regress().to(device)
# model2 = BrainLMEmbeddings(config).to(device)

# model2 =BrainLMModel(config).to(device)
# model2 = BrainLMForPretraining(config).to(device)
model2 =BrainLMDecoder_mask(config, num_patches=196).to(device)



optimizer1 = torch.optim.Adam(model1.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)
optimizer2 = torch.optim.Adam(model2.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)



scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=opt.stepsize, gamma=opt.gamma)
scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=opt.stepsize, gamma=opt.gamma)

############################### Define Other Loss Functions ########################################

loss_fn = nn.MSELoss(reduction='none')
loss_rec = nn.L1Loss(reduction='none')


###################### Network Training Function#####################################
# def train1(epoch):
#     # print('train...........')
#     scheduler1.step()
#
#     model1.train()
#
#     loss_all = 0
#     step = 0
#     for data in train_loader:
#         data = data.to(device)
#         optimizer1.zero_grad()
#         data.y = data.y[:,:28]
#         score_predict , _, _ = model1(data.x.view(int(data.x.shape[0]/opt.nroi),opt.nroi, opt.nroi+9)[:,:,:333], text_feature) #
#
#         data.y = torch.tensor(data.y, dtype=torch.float)
#         column_losses = loss_fn(torch.stack(score_predict)[:,:,0].T, data.y)
#         # 按列求和或取平均
#         loss_c = torch.sum(column_losses)
#
#         loss = opt.lamb0*loss_c #+ opt.lamb1 * correlation_loss #+ opt.lamb2 * loss_p2 \
#         step = step + 1
#
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(parameters=model1.parameters(), max_norm=4.0)
#         loss_all += loss.item() * data.num_graphs
#         optimizer1.step()
#
#
#
#     return loss_all / len(train_dataset) #, s1_arr, s2_arr ,w1,w2




def train2(epoch, train_loader_stage2):
    scheduler2.step()

    model2.train()

    loss_all = 0
    step = 0
    for data in train_loader_stage2:
        data = data.to(device)
        optimizer1.zero_grad()

        fusion_feature_sbj = data.x.view(opt.batchSize, 315, 1024)
        # label_sbj = data.y.view( opt.batchSize, 41)       #28
        merged_matrix = torch.cat([fusion_feature_sbj, regress_weight_tr_repeat], dim=1)
        odd_columns = torch.arange(1, merged_matrix.size(1), 2)
        merged_matrix[:, odd_columns, :] = fusion_feature_sbj
        even_columns = torch.arange(0, merged_matrix.size(1), 2)
        merged_matrix[:, even_columns, :] = regress_weight_tr_repeat
        # merged_matrix = merged_matrix.view(merged_matrix.shape[0], 1, merged_matrix.shape[1], merged_matrix.shape[2])

        # merged_matrix = torch.cat([regress_weight_tr_repeat, regress_weight_tr_repeat], dim=1)
        # xyz_vectors = merged_matrix[:,:,0:3]
        # noise = torch.rand(opt.batchSize, 56, device=device)

        # out = model2(merged_matrix, xyz_vectors, noise)


        out, mask = model2(merged_matrix, model2.training, 0)
        # loss11 = abs((out.logits[:, :, 0, :]-merged_matrix)).sum() *0.01
        # np.corrcoef(out.logits[0, 0, 0, :].detach().cpu().numpy(), merged_matrix[0, 0, :].detach().cpu().numpy())
        t1 = mask.view(opt.batchSize,315*2,1)
        t1 = t1.unsqueeze(-1).repeat(1, 1, 1, out.logits.shape[-1])
        # torch.matmul(merged_matrix[0, 17, :].T, merged_matrix[0, 16, :])    torch.matmul(merged_matrix[0, 17, :].T, out.logits[0, 16, 0, :])
        loss12 = abs((out.logits - merged_matrix.view(opt.batchSize,315*2,1,1024)) * t1).sum()
        loss11 = abs((out.logits[:, :, 0, :]-merged_matrix)).sum() *0.01


        #, xyz_vectors, noise
        # loss11 = out.loss
        # print(loss11)

        loss = opt.lamb0*loss11 +loss12#+ opt.lamb1 * correlation_loss #+ opt.lamb2 * loss_p2 \
        step = step + 1
        print(step)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model2.parameters(), max_norm=4.0)
        loss_all += loss.item() * data.num_graphs
        optimizer2.step()


    return loss_all / len(train_dataset) #, s1_arr, s2_arr ,w1,w2        torch.nn.utils.clip_grad_norm_(parameters=model2.parameters(), max_norm=4.0)









###################### Network Testing Function#####################################



###############################test_acc2

def test_acc2(train_loader_stage2):
    model2.eval()
    correct = []
    device_cpu = torch.device('cpu')

    with torch.no_grad():
      label_train1_corr  =  torch.tensor(torch.zeros(1, text_feature.shape[0]), device=device_cpu)
      label_sys_corr =  torch.tensor(torch.zeros(1, text_feature.shape[0]), device=device_cpu)
      train1_sys_corr =   torch.tensor(torch.zeros(1, text_feature.shape[0]), device=device_cpu)

      label_score  =  torch.tensor(torch.zeros(1, text_feature.shape[0]), device=device_cpu)
      train1_score =  torch.tensor(torch.zeros(1, text_feature.shape[0]), device=device_cpu)
      sys_score =   torch.tensor(torch.zeros(1, text_feature.shape[0]), device=device_cpu)

      for data in train_loader_stage2:
          data = data.to(device)
          optimizer1.zero_grad()

          fusion_feature_sbj = data.x.view(opt.batchSize, 315, 1024)
          label_sbj = data.y.view(opt.batchSize, 315)  #28
          # label_sbj = label_sbj[:,:24]
          merged_matrix = torch.cat([fusion_feature_sbj, regress_weight_tr_repeat], dim=1)
          odd_columns = torch.arange(1, merged_matrix.size(1), 2)
          merged_matrix[:, odd_columns, :] = fusion_feature_sbj
          even_columns = torch.arange(0, merged_matrix.size(1), 2)
          merged_matrix[:, even_columns, :] = regress_weight_tr_repeat
          label_score = torch.cat((label_score, torch.tensor(label_sbj, device=device_cpu)), dim=0)

          # out, mask = model2(merged_matrix, model2.training, 0)
          train1_score_tmp = torch.zeros(label_sbj.shape, device=device)
          sys_score_tmp = torch.zeros(label_sbj.shape, device=device)

          for mask_index in range(regress_weight_tr_repeat.shape[1]):
              out, mask = model2(merged_matrix, model2.training, mask_index*2)
              train1_score_tmp[:,mask_index] = torch.matmul(merged_matrix[:, mask_index * 2 + 1, :], merged_matrix[:, mask_index * 2, :].T)[:, 0]
              sys_score_tmp[:,mask_index] = torch.matmul(merged_matrix[:, mask_index * 2 + 1, :], out.logits[:,mask_index * 2, 0, :].T)[:, 0]
              # np.corrcoef(label_sbj[:, 0].detach().cpu().numpy(), train1_score_tmp[:, 0].detach().cpu().numpy())

          train1_score = torch.cat((train1_score, torch.tensor(train1_score_tmp, device=device_cpu)), dim=0)
          sys_score = torch.cat((sys_score, torch.tensor(sys_score_tmp, device=device_cpu)), dim=0)



    for i in range(label_train1_corr.shape[1]):
        label_train1_corr[0,i] = np.corrcoef(label_score[1:,i].detach().cpu().numpy(), train1_score[1:,i].detach().cpu().numpy())[0,1]
        label_sys_corr[0, i] = np.corrcoef(label_score[1:, i].detach().cpu().numpy(), sys_score[1:, i].detach().cpu().numpy())[0, 1]
        train1_sys_corr[0, i] = np.corrcoef(train1_score[1:, i].detach().cpu().numpy(), sys_score[1:, i].detach().cpu().numpy())[0, 1]



    # if pred_score.shape[0] != 2586:
    #     print(correct[0])
    #     np.save('vit_aug_sink_lossop_' +str(epoch)+ '.npy', np.array(correct))

    # label_score
    # correct = np.sum(correct)/pred_score.shape[1]
    # regress_weight = torch.squeeze(torch.stack(regress_weight), axis=1)
    # regress_weight1 = regress_weight.cpu()
    # del label_score

    return label_train1_corr.mean(),  label_sys_corr.mean(), train1_sys_corr.mean()#correct,  correct, fusion_feature[1:,:,:],  regress_weight1 #/ len(loader.dataset)


def test_acc_zero(train_loader_stage2):
    model2.eval()
    correct = []
    device_cpu = torch.device('cpu')

    with torch.no_grad():
      label_sys_corr =  torch.tensor(torch.zeros(1, hcp_feature.shape[1]), device=device_cpu)
      label_score  =  torch.tensor(torch.zeros(1, hcp_feature.shape[1]), device=device_cpu)
      sys_score =   torch.tensor(torch.zeros(1, hcp_feature.shape[1]), device=device_cpu)

      for data in train_loader_stage2:
          data = data.to(device)
          optimizer2.zero_grad()

          fusion_feature_sbj = data.x.view(opt.batchSize, 109, 1024)
          label_sbj = data.y.view(opt.batchSize, 109)
          fusion_feature_template =fusion_feature_tr[:8, :].to('cuda:0')
          mask_index = 0
          sys_score_tmp = torch.zeros(label_sbj.shape, device=device)

          for score_index in range(label_sbj.shape[1]):
              fusion_feature_template[:, 0:1, :] = fusion_feature_sbj[:, score_index:score_index + 1, :].to('cuda:0')
              merged_matrix = torch.cat([fusion_feature_template, regress_weight_tr_repeat], dim=1)
              odd_columns = torch.arange(1, merged_matrix.size(1), 2)
              merged_matrix[:, odd_columns, :] = fusion_feature_template
              even_columns = torch.arange(0, merged_matrix.size(1), 2)
              merged_matrix[:, even_columns, :] = regress_weight_tr_repeat
              out, mask = model2(merged_matrix, model2.training, mask_index * 2)
              sys_score_tmp[:,score_index] = torch.matmul(merged_matrix[:, mask_index * 2 + 1, :], out.logits[:,mask_index * 2, 0, :].T)[:, 0]



          sys_score = torch.cat((sys_score, torch.tensor(sys_score_tmp, device=device_cpu)), dim=0)
          label_score = torch.cat((label_score, torch.tensor(label_sbj, device=device_cpu)), dim=0)

      for i in range(label_sys_corr.shape[1]):
              label_sys_corr[0, i] = \
              np.corrcoef(label_score[1:, i].detach().cpu().numpy(), sys_score[1:, i].detach().cpu().numpy())[0, 1]
    #
    # np.save('sys_score.npy', sys_score)
    # np.save('label_score.npy', label_score)


    out_cor = torch.mean(label_sys_corr)
    return out_cor, label_sys_corr, sys_score, label_score










def test_loss1(loader,epoch):
    # print('testing...........')
    model1.eval()
    loss_all = 0
    loss_c = []
    for data in loader:
        data = data.to(device)
        data.y = data.y[:, :24]
        score_predict , _ , _ = model1(data.x.view(int(data.x.shape[0]/opt.nroi),opt.nroi, opt.nroi+9)[:,:,:333], text_feature)   #
        data.y = torch.tensor(data.y, dtype=torch.float)
        column_losses = loss_fn(torch.stack(score_predict)[:,:,0].T, data.y)
        # 按列求和或取平均
        loss_c = torch.sum(column_losses)
        loss = opt.lamb0*loss_c #+ opt.lamb1 * correlation_loss #+ opt.lamb2 * loss_p2 \
        loss_all += loss.item() * data.num_graphs
    return loss_all / len(loader.dataset)

############################   Model Training #########################################
best_model_wts = copy.deepcopy(model1.state_dict())
best_loss = 1e10
val_acc_list = []


# for epoch in range(0,stage1_epoch):
#     since  = time.time()
#     # tr_loss, s1_arr, s2_arr, w1, w2 = train(epoch)
#     tr_loss= train1(epoch)
#
#
#     tr_acc, tr_rmse, fusion_feature_tr,  regress_weight_tr, label_score_tr, pred_score_tr= test_acc1(train_loader)
#     val_acc, val_rmse, fusion_feature_val,  regress_weight_val, label_score_val, pred_score_val = test_acc1(val_loader)
#     val_acc_list.append(val_acc)
#     val_loss = test_loss1(val_loader,epoch)
#     time_elapsed = time.time() - since
#     time_elapsed = time.time() - since
#     print('*====**')
#     print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#     print('Epoch: {:03d}, Train Loss: {:.7f}, '
#           'Train Acc: {:.7f},Train rmse: {:.7f}, Test Loss: {:.7f}, Test Acc: {:.7f}, Test rmse: {:.7f}, '.format(epoch, tr_loss,
#                                                        tr_acc, tr_rmse, val_loss, val_acc, val_rmse))

# torch.save(model1, './model/stage1_fmri_best_0602.pth')

# torch.save([train_loader, val_loader], './model/stage1_dataset_0602.pt')

#########################################################
fusion_feature = fusion_feature_tr
fusion_feature = fusion_feature.view(fusion_feature.shape[0], fusion_feature.shape[1]*fusion_feature.shape[2])
train_dataset_stage2 = CustomDataset(fusion_feature, label_score_tr)
train_loader_stage2 = DataLoader(train_dataset_stage2, batch_size=opt.batchSize, shuffle=False, drop_last=True)
# torch.matmul(fusion_feature_tr[0,0,:].T, regress_weight_tr[0,:])





fusion_feature = hcp_feature
fusion_feature = fusion_feature.view(fusion_feature.shape[0], fusion_feature.shape[1]*fusion_feature.shape[2])
val_dataset_stage2 = CustomDataset(fusion_feature, hcp_label)
val_dataset_stage2 = DataLoader(val_dataset_stage2, batch_size=opt.batchSize, shuffle=True, drop_last=True)


regress_weight_tr_repeat = regress_weight_tr.unsqueeze(0).repeat(opt.batchSize, 1, 1).to(device)

# torch.save([train_loader_stage2, val_dataset_stage2, regress_weight_tr_repeat], './model/stage1_dataset.pt')

#####################################################################


for epoch in range(stage1_epoch, 1000):
    since  = time.time()

    tr_loss=  train2(epoch, train_loader_stage2)
    label_sys_corr_val_mean, label_sys_corr , sys_score, label_score = test_acc_zero(val_dataset_stage2)    #   test_acc2(val_dataset_stage2)

    filename111 = f'label_sys_corr_epoch_{epoch}.npy'
    np.save(filename111, label_sys_corr)

    filename111 = f'sys_score_epoch_{epoch}.npy'
    np.save(filename111, sys_score)

    filename111 = f'label_score_epoch_{epoch}.npy'
    np.save(filename111, label_score)

    ##label_sys_corr_mean,


    time_elapsed = time.time() - since


    print('*====**')
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Epoch: {:03d}, Train Loss: {:.7f}, '.format(epoch, tr_loss))



    print('*====**')
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Epoch: {:03d}, Train Loss: {:.7f}, label_sys_corr_mean: {:.7f}'.format(epoch, tr_loss,  label_sys_corr_val_mean))








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
   # test_accuracy , test_rmse= test_acc1(test_loader)
   # test_l= test_loss1(test_loader,0)
   print("===========================")
   # print("Test Acc: {:.7f}, Test Loss: {:.7f} ".format(test_accuracy, test_l))
   print(opt)

