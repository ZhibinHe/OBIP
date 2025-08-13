import os
import numpy as np
import argparse
import time
import copy

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

from imports.HCPDataset import HCPfmriScoreDataset_sbjnum, ABCDfmriScoreDataset_sbjnum


from torch_geometric.data import DataLoader
from net.braingnn import Network_regress_score
from imports.utils import train_val_test_split_hcp
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from imports.vit import ViT
from net.Network_Dual_ViT import *
from net.Network_Pred import *
from net.Network_Combine_HCP import *



import pandas as pd
import pylab


# 333 node Epoch: 018 0.1822354,  0m 7s



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

# bs=8, lamb0=0, lamb1=100, epoch=37, 0.2030

# bs=50, lamb0=0, lamb1=100, epoch=37, 0.1664





opt = parser.parse_args()

if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path)

#################### Parameter Initialization #######################
path = opt.dataroot
name = 'HCP'
save_model = opt.save_model
load_model = opt.load_model
opt_method = opt.optim
num_epoch = opt.n_epochs
fold = opt.fold
writer = SummaryWriter(os.path.join('./log',str(fold)))



################## Define Dataloader ##################################

# dataset = HCPfmriScoreDataset_sbjnum(path,name)

text_feature = torch.load(os.path.join(opt.datadir+'/ABCD_phenotype_feature_1681_1110.pt'))
text_feature = text_feature[0:315,:]
dataset = HCPfmriScoreDataset_sbjnum(path,name)
csvdata = pd.read_csv(os.path.join(opt.csvroot+'/ABCD_PhenotypeABCD_Phenotype_sbjid_315.csv'))
traincsvdata = pd.read_csv(os.path.join(opt.csvroot+'/ABCD_Phenotype_Description_315.csv'))



csvdata_head = csvdata.columns.tolist()
select_score  = np.zeros((csvdata.shape[1]-1, csvdata.shape[0]))

for i in range(traincsvdata['var_name'].shape[0]):
    select_score[i] = csvdata[traincsvdata['var_name'][i]]


# non_nan_columns = np.where(~np.isnan(select_score).any(axis=0))[0]
# select_score_1 = select_score[:, non_nan_columns]
# csv_fname_values = csvdata["Subject"][non_nan_columns].values


select_score_1 = select_score
csv_fname_values = csvdata["NAME"].values




dataset.data.sbj_fname = [(x) for x in dataset.data.sbj_fname]
dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)


###score

csv_indices = np.where(select_fname == csv_fname_values[ :, np.newaxis])
select_score_2 = select_score_1[:,csv_indices[0]]

#
for i in range(select_score_2.shape[0]):
    select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]]- np.mean(select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])



y_arr = select_score_2
y_arr = y_arr.T
y_torch = torch.from_numpy(y_arr)

# (select_score_2[6,:] - np.mean(select_score_2[6,:])) / np.std(select_score_2[6,:])    select_score_2[6,np.where(~np.isnan(select_score_2[6,:]))[0]]

###other
dataset_indices = np.where(select_fname == dataset_sbj_fname[ :, np.newaxis])

dataset_x = np.reshape(dataset.data.x, (9438, 400, 400))
dataset_x = dataset_x[dataset_indices[0],:,:]
dataset_x = np.reshape(dataset_x, (dataset_x.shape[0]*dataset_x.shape[1], 400))

dataset_pos = np.reshape(dataset.data.pos, (9438, 400, 400))
dataset_pos = dataset_pos[dataset_indices[0],:,:]
dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0]*dataset_pos.shape[1], 400))


dataset_indices_torch = torch.from_numpy(dataset_indices[0])

# att_indices = np.where(dataset_indices_torch == dataset.data.edge_sbj_torch[ :, np.newaxis])

att_indices = np.arange(0,dataset.data.edge_sbj_torch[ :, np.newaxis].shape[0])
dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0],:]
dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
dataset_edge_attr = dataset.data.edge_attr[att_indices[0],:]

#########################

dataset.data.x = dataset_x
dataset.data.edge_index = dataset_edge_index
dataset.data.edge_attr = dataset_edge_attr
dataset.data.y = y_torch
dataset.data.pos = dataset_pos
dataset.data.edge_sbj_torch = dataset_edge_sbj_torch
dataset.data.sbj_fname = select_fname


del dataset.data.edge_index
del dataset.data.edge_attr
del dataset.data.pos
del dataset.data.edge_sbj_torch
del dataset.data.sbj_fname
# torch.where(dataset.data.edge_sbj_torch!=dataset_indices)
#
# dataset.data.sbj_fname[0]
# csvdata["Emotion_Task_Acc"][non_nan_index]
# csvdata["Subject"]



# dataset.data.y = dataset.data.y.squeeze()


dataset.data.x[dataset.data.x == float('inf')] = 0

tr_index,val_index,te_index = train_val_test_split_hcp(n_sub = dataset.data.y.size()[0], fold=fold)
tr_index = np.concatenate((tr_index, te_index))

# tr_index = tr_index[:-(tr_index.shape[0]%10)]
# val_index = val_index[:-(val_index.shape[0]%10)]

train_dataset = dataset[tr_index]
val_dataset = dataset[val_index]
test_dataset = dataset[val_index]


train_loader = DataLoader(train_dataset,batch_size=opt.batchSize, shuffle= True)
val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)

# np.save('y_arr_val.npy',val_dataset.data.y.detach().cpu().numpy())

############### Define Graph Deep Learning Network ##########################

model = CombinedModel_sink400().to(device)






# print(model)

if opt_method == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)
elif opt_method == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr =opt.lr, momentum = 0.9, weight_decay=opt.weightdecay, nesterov = True)

scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)

############################### Define Other Loss Functions ########################################

loss_fn = nn.MSELoss(reduction='none')


###################### Network Training Function#####################################
def train(epoch):
    # print('train...........')
    scheduler.step()

    model.train()

    loss_all = 0
    step = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        score_predict , regress_weight= model(data.x.view(int(data.x.shape[0]/opt.nroi),opt.nroi, opt.nroi), text_feature) #
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
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=4.0)
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
        print(step)
        # print(loss_all / len(train_dataset))
        # s1_arr = np.hstack(s1_list)
        # s2_arr = np.hstack(s2_list)
    return loss_all / len(train_dataset) #, s1_arr, s2_arr ,w1,w2


###################### Network Testing Function#####################################
def test_acc(loader):
    model.eval()
    correct = []


    with torch.no_grad():
      pred_score = torch.tensor(torch.zeros(1, text_feature.shape[0]), device=device)
      label_score = torch.tensor(torch.zeros(1, text_feature.shape[0]), device=device)
      for data in loader:
        data = data.to(device)
        # outputs = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
        score_predict , regress_weight = model(data.x.view(int(data.x.shape[0]/opt.nroi),opt.nroi, opt.nroi), text_feature)  #
        score_predict1 = torch.stack(score_predict)[:, :, 0].T
        pred_score = torch.cat((pred_score, score_predict1), dim=0)
        label_score = torch.cat((label_score, data.y), dim=0)

    label_score[torch.where(torch.isnan(label_score) == True)] = 0
    for i in range(pred_score.shape[1]):
        correct_task = np.corrcoef(pred_score[1:, i].detach().cpu().numpy().T, label_score[1:,i].detach().cpu().numpy().T)[0, 1]
        correct.append(correct_task)

    correct1 = np.sum(correct)/pred_score.shape[1]



    return correct1,  correct1, correct   #/ len(loader.dataset)

def test_loss(loader,epoch):
    # print('testing...........')
    model.eval()
    loss_all = 0
    loss_c = []
    for data in loader:
        data = data.to(device)

        score_predict , regress_weight = model(data.x.view(int(data.x.shape[0]/opt.nroi),opt.nroi, opt.nroi), text_feature)   #




        data.y = torch.tensor(data.y, dtype=torch.float)
        data.y[torch.where(torch.isnan(data.y) == True)] = 0

        nan_mask = torch.isnan(data.y)
        isnan_matrix = torch.zeros_like(data.y)
        isnan_matrix[~nan_mask] = 1
        # torch.stack(score_predict)

        # column_losses = loss_fn(torch.stack(score_predict)[:,:,0].T , data.y )


        column_losses = loss_fn(torch.stack(score_predict)[:,:,0].T *isnan_matrix, data.y * isnan_matrix)

        # 按列求和或取平均
        loss_c = torch.sum(column_losses)

        correlation = F.cosine_similarity(torch.stack(score_predict)[:,:,0].T, data.y, dim=0)
        #
        # # 计算相关性损失
        correlation_loss = 1 - correlation.mean()

        loss = opt.lamb0*loss_c + opt.lamb1 * correlation_loss #+ opt.lamb2 * loss_p2 \







        loss_all += loss.item() * data.num_graphs
    return loss_all / len(loader.dataset)

#######################################################################################
############################   Model Training #########################################
#######################################################################################
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 1e10
val_acc_list = []
for epoch in range(0, num_epoch):
    since  = time.time()
    # tr_loss, s1_arr, s2_arr, w1, w2 = train(epoch)
    tr_loss= train(epoch)

    # tr_acc, tr_rmse, tr_acc_sbj = test_acc(train_loader)
    val_acc, val_rmse, val_acc_sbj = test_acc(val_loader)
    tr_acc = val_acc
    tr_rmse = val_rmse


    filename111 = f'val_acc_315_epoch_{epoch}.npy'
    np.save(filename111, val_acc_sbj)


    val_acc_list.append(val_acc)


    val_loss = test_loss(val_loader,epoch)
    time_elapsed = time.time() - since
    print('*====**')
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Epoch: {:03d}, Train Loss: {:.7f}, '
          'Train Acc: {:.7f},Train rmse: {:.7f}, Test Loss: {:.7f}, Test Acc: {:.7f}, Test rmse: {:.7f}, '.format(epoch, tr_loss,
                                                       tr_acc, tr_rmse, val_loss, val_acc, val_rmse))


    writer.add_scalars('Acc',{'train_acc':tr_acc,'val_acc':val_acc},  epoch)
    writer.add_scalars('Loss', {'train_loss': tr_loss, 'val_loss': val_loss},  epoch)


#######################################################################################
######################### Testing on testing set ######################################
#######################################################################################

if opt.load_model:
    model = Network_regress_score(opt.indim,opt.ratio,opt.nclass).to(device)
    model.load_state_dict(torch.load(os.path.join(opt.save_path,str(fold)+'.pth')))
    model.eval()
    preds = []
    correct = 0
    for data in val_loader:
        data = data.to(device)
        outputs= model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos)
        pred = outputs[0].max(1)[1]
        preds.append(pred.cpu().detach().numpy())
        correct += pred.eq(data.y).sum().item()
    preds = np.concatenate(preds,axis=0)
    trues = val_dataset.data.y.cpu().detach().numpy()
    cm = confusion_matrix(trues,preds)
    print("Confusion matrix")
    print(classification_report(trues, preds))

else:
   model.load_state_dict(best_model_wts)
   model.eval()
   test_accuracy , test_rmse= test_acc(test_loader)
   test_l= test_loss(test_loader,0)
   print("===========================")
   print("Test Acc: {:.7f}, Test Loss: {:.7f} ".format(test_accuracy, test_l))
   print(opt)

