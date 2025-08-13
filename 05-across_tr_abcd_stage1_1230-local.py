import os
import numpy as np
import argparse
import time
import copy
import pandas as pd

from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from net.braingnn import Network_regress_score
from sklearn.metrics import classification_report, confusion_matrix
from imports.data_load_hcp import *
from net.configuration_brainlm import BrainLMConfig
from net.Network_Combine_HCP import *
from torch_geometric.data import Data, Dataset, DataLoader
from net.modeling_brainlm import *

#
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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







opt = parser.parse_args()

if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path)

#################### Parameter Initialization #######################
path = opt.dataroot
# path2 = opt.datatoot_hcp
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

text_feature_zero = text_feature_all.repeat(opt.batchSize, 1, 1)


train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)


model1 = CombinedModel_sink_feature400().to(device)     # ABCD 115
model2 =BrainLMDecoder_mask(config, num_patches=196).to(device)



optimizer1 = torch.optim.Adam(model1.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)
optimizer2 = torch.optim.Adam(model2.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)



scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=opt.stepsize, gamma=opt.gamma)
scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=opt.stepsize, gamma=opt.gamma)

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
        # text_feature = text_feature.to(device)

        score_predict , _, _= model1(data.x.view(int(data.x.shape[0]/opt.nroi),opt.nroi, opt.nroi), text_feature.to(device)) #
        data.y = torch.tensor(data.y, dtype=torch.float)


        nan_mask = torch.isnan(data.y)
        isnan_matrix = torch.zeros_like(data.y)
        isnan_matrix[~nan_mask] = 1

        data.y[torch.where(torch.isnan(data.y) == True)] = 0


        column_losses = loss_fn(torch.stack(score_predict)[:,:,0].T *isnan_matrix, data.y * isnan_matrix)
        # 按列求和或取平均
        loss_c = torch.sum(column_losses)
        loss = opt.lamb0*loss_c #+ opt.lamb1 * correlation_loss #+ opt.lamb2 * loss_p2 \
        step = step + 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model1.parameters(), max_norm=4.0)
        loss_all += loss.item() * data.num_graphs
        optimizer1.step()
        # print(step)


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
      # TT=0

      for data in loader:
        data = data.to(device)
        # outputs = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
        score_predict, regress_weight, feature_cat = model1(data.x.view(int(data.x.shape[0]/opt.nroi),opt.nroi, opt.nroi), text_feature.to(device))  #
        score_predict1 = torch.stack(score_predict)[:, :, 0].T
        pred_score = torch.cat((pred_score, score_predict1), dim=0)
        label_score = torch.cat((label_score, data.y), dim=0)
        fusion_feature = torch.cat((fusion_feature, feature_cat.cpu()), dim=0)
        # TT = TT+1
        # print(TT)

    # label_score[torch.where(torch.isnan(label_score) == True)] = 0
    for i in range(pred_score.shape[1]):

        pred_score_tmp =  pred_score[torch.where(torch.isnan(label_score[:,i]) == False)[0],i][1:]
        label_score_tmp = label_score[torch.where(torch.isnan(label_score[:,i]) == False)[0],i][1:]

        correct_task = np.corrcoef(pred_score_tmp.detach().cpu().numpy().T, label_score_tmp.detach().cpu().numpy().T)[0, 1]
        correct.append(correct_task)


        # correct_task = np.corrcoef(pred_score[1:, i].detach().cpu().numpy().T, label_score[1:,i].detach().cpu().numpy().T)[0, 1]
        # correct.append(correct_task)

    correct = np.sum(correct)/pred_score.shape[1]
    regress_weight = torch.squeeze(torch.stack(regress_weight), axis=1)
    regress_weight1 = regress_weight.cpu()
    del regress_weight

    return correct,  correct, fusion_feature[1:,:,:],  regress_weight1, label_score[1:,:], pred_score[1:,:] #/ len(loader.dataset)




def write_epoch_acc_to_file(epoch, acc, file_path='epoch_acc.txt'):
    with open(file_path, 'a') as file:
        file.write(f'Epoch: {epoch}, Accuracy: {acc}\n')






def test_loss1(loader,epoch):
    # print('testing...........')
    model1.eval()
    loss_all = 0
    loss_c = []
    for data in loader:
        data = data.to(device)
        # text_feature = text_feature.to(device)

        score_predict , _, _= model1(data.x.view(int(data.x.shape[0]/opt.nroi),opt.nroi, opt.nroi), text_feature.to(device)) #
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
    tr_loss= train1(epoch)
    tr_acc, tr_rmse, fusion_feature_tr, regress_weight_tr, label_score_tr, pred_score_tr = test_acc1(train_loader)
    val_acc, val_rmse, fusion_feature_val, regress_weight_val, label_score_val, pred_score_val = test_acc1(val_loader)

    # model_filename = f'./model/stage1_across_tr_abcd_model_{epoch}.pth'
    # dataset_filename =  f'./model/stage1_across_tr_abcd_dataset_{epoch}.pt'
    # torch.save(model1, model_filename)
    # torch.save([train_loader, val_loader,text_feature, text_feature_zero, fusion_feature_tr, fusion_feature_val, label_score_tr, label_score_val, regress_weight_tr, regress_weight_val], dataset_filename)



    time_elapsed = time.time() - since
    time_elapsed = time.time() - since

    # write_epoch_acc_to_file(epoch, val_acc, file_path='./model/stage1_across_tr_abcd_epoch_acc.txt')

    print('*====**')
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Epoch: {:03d}, '
          'Test Acc: {:.7f}, '.format(epoch, val_acc))



####################delete#########################
###################################################

#
# df = pd.read_csv('./model/stage1_across_tr_abcd_epoch_acc.txt', sep=r'[,\s]+')
# all_epoch_acc = df.iloc[:, 3].to_numpy()
# all_epoch_epoch = df.iloc[:, 1].to_numpy()
#
# for i in range(all_epoch_acc.shape[0]):
#     if i ==np.argmax(all_epoch_acc):
#         continue
#     model_filename = f'./model/stage1_across_tr_abcd_model_{all_epoch_epoch[i]}.pth'
#     dataset_filename =  f'./model/stage1_across_tr_abcd_dataset_{all_epoch_epoch[i]}.pt'
#
#     if os.path.exists(model_filename):
#         os.remove(model_filename)
#     if os.path.exists(dataset_filename):
#         os.remove(dataset_filename)
#
# model_filename = f'./model/stage1_across_tr_abcd_model_0.pth'
# dataset_filename =  f'./model/stage1_across_tr_abcd_dataset_0.pt'
#
# if os.path.exists(model_filename):
#     os.remove(model_filename)
# if os.path.exists(dataset_filename):
#     os.remove(dataset_filename)
#
#
