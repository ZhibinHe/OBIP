import os
from imports.ABIDEDataset import HCPfmriScoreDataset_sbjnum, HCPT1wScoreDataset_sbjnum, HCPAfmriScoreDataset_sbjnum, HCPAT1wScoreDataset_sbjnum
from torch_geometric.data import DataLoader
from imports.utils import train_val_test_split_hcp
from net.Network_Combine import *
from net.models_mae import *
import pandas as pd
import torch


def data_load_hcp(path, name, opt):
    fold = opt.fold

    text_feature = torch.load(os.path.join(opt.datadir + '/phenotype_text_feature_tr.pt'))
    ##################
    text_feature2 = torch.load(os.path.join(opt.datadir + '/phenotype_text_feature_te.pt'))
    text_feature = torch.cat((text_feature, text_feature2), dim=1)
########################
    dataset = HCPfmriScoreDataset_sbjnum(path, name)
    # dataset2 = HCPT1wScoreDataset_sbjnum(path, name)
    csvdata = pd.read_csv(os.path.join(opt.csvroot + '/S900 Release Subjects 4rsfmri.csv'))

    # traincsvdata = pd.read_csv(os.path.join(opt.datadir + '/HCP_train_phenotype.csv'))
    traincsvdata = pd.read_csv(os.path.join(opt.datadir + '/HCP_all_phenotype.csv'))

    select_score = np.zeros((traincsvdata['0'].shape[0], csvdata.shape[0]))

    for i in range(traincsvdata['0'].shape[0]):
        select_score[i] = csvdata[traincsvdata['0'][i]]

    non_nan_columns = np.where(~np.isnan(select_score).any(axis=0))[0]
    select_score_1 = select_score[:, non_nan_columns]

    csv_fname_values = csvdata["rsfmrilist"][non_nan_columns].values

    dataset.data.sbj_fname = [str(x) for x in dataset.data.sbj_fname]

    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)
    ###score
    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]

    y_arr = (select_score_2 - np.mean(select_score_2, axis=1, keepdims=True)) / np.std(select_score_2, axis=1,
                                                                                       keepdims=True)
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    ###other
    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])
    dataset_x = np.reshape(dataset.data.x, (3427, 333, 333))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 333))

    #########################
    dataset.data.x = dataset_x
    dataset.data.y = y_torch
    dataset.data.sbj_fname = select_fname

    del dataset.data.edge_index
    del dataset.data.edge_attr
    del dataset.data.pos
    del dataset.data.edge_sbj_torch

    dataset.data.x[dataset.data.x == float('inf')] = 0

    for i in range(select_fname.size):
        select_fname[i] = select_fname[i][:-5]
    select_fname2 = np.unique(select_fname)
    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=select_fname2.size, fold=fold)
    tr_index = np.concatenate((tr_index, te_index))

    # test_index1 = list()
    train_index1 = list()
    val_index1 = list()

    for tr in tr_index:
        train_index1 = np.concatenate((train_index1, np.where(select_fname2[tr] == select_fname)[0]))
    train_index1 = train_index1.astype(np.int64)

    for val in val_index:
        val_index1 = np.concatenate((val_index1, np.where(select_fname2[val] == select_fname)[0]))
    val_index1 = val_index1.astype(np.int64)



    train_dataset = dataset[train_index1]
    val_dataset = dataset[val_index1]
    test_dataset = dataset[val_index1]


    return train_dataset, val_dataset, test_dataset, text_feature, dataset, train_index1, val_index1



def data_load_hcp_t1fmri(path, path2, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    fold = opt.fold

    text_feature = torch.load(os.path.join(opt.datadir + '/phenotype_text_feature_tr.pt'), map_location=device)
    ##################
    text_feature2 = torch.load(os.path.join(opt.datadir + '/phenotype_text_feature_te.pt'), map_location=device)
    text_feature = torch.cat((text_feature, text_feature2), dim=1)
########################
    dataset = HCPfmriScoreDataset_sbjnum(path, name)
    dataset2 = HCPT1wScoreDataset_sbjnum(path2, name)

    multi_modal_data = torch.zeros(dataset.data.y.size()[0],dataset.data.x.size()[1],dataset.data.x.size()[1]+dataset2.data.x.size()[1])
    fmri_data = torch.reshape(dataset.data.x, (dataset.data.y.size()[0], dataset.data.x.size()[1],dataset.data.x.size()[1]))
    t1w_data = torch.reshape(dataset2.data.x, (dataset2.data.y.size()[0], dataset.data.x.size()[1],dataset2.data.x.size()[1]))

    delete_index = []

    multi_modal_data[:, :, 0:fmri_data.size()[1]] = fmri_data
    for i in range(fmri_data.size()[0]):
        try:
            my_index = dataset2.data.sbj_fname.index(dataset.data.sbj_fname[i][0:6])
            multi_modal_data[i, :, fmri_data.size()[1]:] = t1w_data[my_index, :, :]
        except:
            delete_index.append(i)
            # print(dataset.data.sbj_fname[i])

    fmri_data =  multi_modal_data.index_select(0, torch.tensor([i for i in range(multi_modal_data.size(0)) if i not in torch.tensor(delete_index)]))


    dataset.data.x = torch.reshape(fmri_data, (fmri_data.size(0) * fmri_data.size(1), 333+9))
    dataset.data.sbj_fname = [row for i, row in enumerate(dataset.data.sbj_fname) if i not in delete_index]



    csvdata = pd.read_csv(os.path.join(opt.csvroot + '/S900 Release Subjects 4rsfmrit1w.csv'))

    # traincsvdata = pd.read_csv(os.path.join(opt.datadir + '/HCP_train_phenotype.csv'))
    traincsvdata = pd.read_csv(os.path.join(opt.datadir + '/HCP_all_phenotype.csv'))

    select_score = np.zeros((traincsvdata['0'].shape[0], csvdata.shape[0]))

    for i in range(traincsvdata['0'].shape[0]):
        select_score[i] = csvdata[traincsvdata['0'][i]]

    # select_score = np.delete(select_score, delete_index, axis=1)

    non_nan_columns = np.where(~np.isnan(select_score).any(axis=0))[0]
    select_score_1 = select_score[:, non_nan_columns]

    # csvdata["rsfmrilist"] = csvdata["rsfmrilist"].drop(csvdata["rsfmrilist"].index[delete_index])



    csv_fname_values = csvdata["rsfmrilist"][non_nan_columns].values

    dataset.data.sbj_fname = [str(x) for x in dataset.data.sbj_fname]

    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)
    ###score
    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]

    y_arr = (select_score_2 - np.mean(select_score_2, axis=1, keepdims=True)) / np.std(select_score_2, axis=1,
                                                                                       keepdims=True)
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    ###other
    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])
    dataset_x = np.reshape(dataset.data.x, (dataset_sbj_fname.size, 333, 333+9))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 333+9))

    #########################
    dataset.data.x = dataset_x
    dataset.data.y = y_torch
    dataset.data.sbj_fname = select_fname

    del dataset.data.edge_index
    del dataset.data.edge_attr
    del dataset.data.pos
    del dataset.data.edge_sbj_torch

    dataset.data.x[dataset.data.x == float('inf')] = 0

    for i in range(select_fname.size):
        select_fname[i] = select_fname[i][:-5]
    select_fname2 = np.unique(select_fname)
    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=select_fname2.size, fold=fold)
    tr_index = np.concatenate((tr_index, te_index))

    # test_index1 = list()
    train_index1 = list()
    val_index1 = list()

    for tr in tr_index:
        train_index1 = np.concatenate((train_index1, np.where(select_fname2[tr] == select_fname)[0]))
    train_index1 = train_index1.astype(np.int64)

    for val in val_index:
        val_index1 = np.concatenate((val_index1, np.where(select_fname2[val] == select_fname)[0]))
    val_index1 = val_index1.astype(np.int64)



    train_dataset = dataset[train_index1]
    val_dataset = dataset[val_index1]
    test_dataset = dataset[val_index1]


    return train_dataset, val_dataset, test_dataset, text_feature, dataset, train_index1, val_index1



def data_load_hcpa_t1fmri(path, path2, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    fold = opt.fold

    text_feature = torch.load(os.path.join(opt.datadir + '/HCPA_phenotype_text_feature_tr.pt'), map_location=device)
    ##################
    text_feature2 = torch.load(os.path.join(opt.datadir + '/HCPA_phenotype_text_feature_te.pt'), map_location=device)
    text_feature = torch.cat((text_feature, text_feature2), dim=1)
########################
    dataset = HCPAfmriScoreDataset_sbjnum(path, name)
    dataset2 = HCPAT1wScoreDataset_sbjnum(path2, name)

    multi_modal_data = torch.zeros(dataset.data.y.size()[0],dataset.data.x.size()[1],dataset.data.x.size()[1]+dataset2.data.x.size()[1])
    fmri_data = torch.reshape(dataset.data.x, (dataset.data.y.size()[0], dataset.data.x.size()[1],dataset.data.x.size()[1]))
    t1w_data = torch.reshape(dataset2.data.x, (dataset2.data.y.size()[0], dataset.data.x.size()[1],dataset2.data.x.size()[1]))

    delete_index = []

    multi_modal_data[:, :, 0:fmri_data.size()[1]] = fmri_data
    for i in range(fmri_data.size()[0]):
        try:
            my_index = dataset2.data.sbj_fname.index(dataset.data.sbj_fname[i][:-5])
            multi_modal_data[i, :, fmri_data.size()[1]:] = t1w_data[my_index, :, :]
        except:
            delete_index.append(i)
            # print(dataset.data.sbj_fname[i])

    fmri_data =  multi_modal_data.index_select(0, torch.tensor([i for i in range(multi_modal_data.size(0)) if i not in torch.tensor(delete_index)]))


    dataset.data.x = torch.reshape(fmri_data, (fmri_data.size(0) * fmri_data.size(1), 333+9))
    dataset.data.sbj_fname = [row for i, row in enumerate(dataset.data.sbj_fname) if i not in delete_index]



    csvdata = pd.read_csv(os.path.join(opt.csvroot + '/HCP_Aging_phenotype-select.csv'))

    # traincsvdata = pd.read_csv(os.path.join(opt.datadir + '/HCP_train_phenotype.csv'))
    traincsvdata = pd.read_csv(os.path.join(opt.datadir + '/HCPA_all_phenotype .csv'))

    select_score = np.zeros((traincsvdata['0'].shape[0], csvdata.shape[0]))

    for i in range(traincsvdata['0'].shape[0]):
        select_score[i] = csvdata[traincsvdata['0'][i]]

    # select_score = np.delete(select_score, delete_index, axis=1)

    non_nan_columns = np.where(~np.isnan(select_score).any(axis=0))[0]
    select_score_1 = select_score[:, non_nan_columns]

    # csvdata["rsfmrilist"] = csvdata["rsfmrilist"].drop(csvdata["rsfmrilist"].index[delete_index])



    csv_fname_values = csvdata["subject_id_fmri"][non_nan_columns].values

    dataset.data.sbj_fname = [str(x) for x in dataset.data.sbj_fname]

    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)
    ###score
    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]

    y_arr = (select_score_2 - np.mean(select_score_2, axis=1, keepdims=True)) / np.std(select_score_2, axis=1,
                                                                                       keepdims=True)
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    ###other
    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])
    dataset_x = np.reshape(dataset.data.x, (dataset_sbj_fname.size, 333, 333+9))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 333+9))

    #########################
    dataset.data.x = dataset_x
    dataset.data.y = y_torch
    dataset.data.sbj_fname = select_fname

    del dataset.data.edge_index
    del dataset.data.edge_attr
    del dataset.data.pos
    del dataset.data.edge_sbj_torch

    dataset.data.x[dataset.data.x == float('inf')] = 0

    for i in range(select_fname.size):
        select_fname[i] = select_fname[i][:-5]
    select_fname2 = np.unique(select_fname)
    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=select_fname2.size, fold=fold)
    tr_index = np.concatenate((tr_index, te_index))

    # test_index1 = list()
    train_index1 = list()
    val_index1 = list()

    for tr in tr_index:
        train_index1 = np.concatenate((train_index1, np.where(select_fname2[tr] == select_fname)[0]))
    train_index1 = train_index1.astype(np.int64)

    for val in val_index:
        val_index1 = np.concatenate((val_index1, np.where(select_fname2[val] == select_fname)[0]))
    val_index1 = val_index1.astype(np.int64)



    train_dataset = dataset[train_index1]
    val_dataset = dataset[val_index1]
    test_dataset = dataset[val_index1]


    return train_dataset, val_dataset, test_dataset, text_feature, dataset, train_index1, val_index1





def data_load_abcd_hcp_d_hcp_ya_fmri(path, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    fold = opt.fold


    text_feature = torch.load(os.path.join(opt.phenoroot + '/0112_ABCD_HCP_D_HCP_YA_Phenotype_Description_184.pt'), map_location=device)
    text_feature = text_feature[0:text_feature.shape[0], :]
    dataset = HCPfmriScoreDataset_sbjnum(path, name)
    # csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/0112_ABCD_HCP_D_HCP_YA_Phenotype_Value_184.csv'))  #pheno_value
    csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/0112_ABCD_HCP_D_HCP_YA_reshape_Phenotype_Value_184.csv'))  #pheno_value

    traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/0112_ABCD_HCP_D_HCP_YA_Phenotype_Description_184.csv')) # description

    # csvdata_head = csvdata.columns.tolist()
    select_score = np.zeros((csvdata.shape[1] - 1, csvdata.shape[0]))

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

     # save dataset_sbj_fname as .csv



    ###score

    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]

    #
    for i in range(select_score_2.shape[0]):
        select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(
            ~np.isnan(select_score_2[i, :]))[0]] - np.mean(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])

    y_arr = select_score_2
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    # (select_score_2[6,:] - np.mean(select_score_2[6,:])) / np.std(select_score_2[6,:])    select_score_2[6,np.where(~np.isnan(select_score_2[6,:]))[0]]

    ###other
    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])

    dataset_x = np.reshape(dataset.data.x, (select_fname.size, 400, 400))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 400))

    dataset_pos = np.reshape(dataset.data.pos, (select_fname.size, 400, 400))
    dataset_pos = dataset_pos[dataset_indices[0], :, :]
    dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0] * dataset_pos.shape[1], 400))

    dataset_indices_torch = torch.from_numpy(dataset_indices[0])

    # att_indices = np.where(dataset_indices_torch == dataset.data.edge_sbj_torch[ :, np.newaxis])

    att_indices = np.arange(0, dataset.data.edge_sbj_torch[:, np.newaxis].shape[0])
    dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0], :]
    dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
    dataset_edge_attr = dataset.data.edge_attr[att_indices[0], :]

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

    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=dataset.data.y.size()[0], fold=fold)
    tr_index = np.concatenate((tr_index, te_index))
    # tr_index = np.concatenate((tr_index, te_index, val_index))

    # tr_index = tr_index[:-(tr_index.shape[0]%10)]
    # val_index = val_index[:-(val_index.shape[0]%10)]

    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[val_index]



    return train_dataset, val_dataset, test_dataset, text_feature #, dataset, train_index1, val_index1

def data_load_abcd_hcp_a_hcp_d_fmri(path, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    fold = opt.fold


    text_feature = torch.load(os.path.join(opt.phenoroot + '/0112_ABCD_HCP_A_HCP_D_Phenotype_Description_183.pt'), map_location=device)
    text_feature = text_feature[0:text_feature.shape[0], :]
    dataset = HCPfmriScoreDataset_sbjnum(path, name)
    csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/0112_ABCD_HCP_A_HCP_D_reshape_Phenotype_Value_183.csv'))  #pheno_value
    traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/0112_ABCD_HCP_A_HCP_D_Phenotype_Description_183.csv')) # description

    # csvdata_head = csvdata.columns.tolist()
    select_score = np.zeros((csvdata.shape[1] - 1, csvdata.shape[0]))

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

     # save dataset_sbj_fname as .csv



    ###score

    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]

    #
    for i in range(select_score_2.shape[0]):
        select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(
            ~np.isnan(select_score_2[i, :]))[0]] - np.mean(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])

    y_arr = select_score_2
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    # (select_score_2[6,:] - np.mean(select_score_2[6,:])) / np.std(select_score_2[6,:])    select_score_2[6,np.where(~np.isnan(select_score_2[6,:]))[0]]

    ###other
    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])

    dataset_x = np.reshape(dataset.data.x, (select_fname.size, 400, 400))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 400))

    dataset_pos = np.reshape(dataset.data.pos, (select_fname.size, 400, 400))
    dataset_pos = dataset_pos[dataset_indices[0], :, :]
    dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0] * dataset_pos.shape[1], 400))

    dataset_indices_torch = torch.from_numpy(dataset_indices[0])

    # att_indices = np.where(dataset_indices_torch == dataset.data.edge_sbj_torch[ :, np.newaxis])

    att_indices = np.arange(0, dataset.data.edge_sbj_torch[:, np.newaxis].shape[0])
    dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0], :]
    dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
    dataset_edge_attr = dataset.data.edge_attr[att_indices[0], :]

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

    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=dataset.data.y.size()[0], fold=fold)
    tr_index = np.concatenate((tr_index, te_index))
    # tr_index = np.concatenate((tr_index, te_index, val_index))

    # tr_index = tr_index[:-(tr_index.shape[0]%10)]
    # val_index = val_index[:-(val_index.shape[0]%10)]

    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[val_index]



    return train_dataset, val_dataset, test_dataset, text_feature #, dataset, train_index1, val_index1


def data_load_abcd_hcp_a_hcp_ya_fmri(path, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    fold = opt.fold


    text_feature = torch.load(os.path.join(opt.phenoroot + '/0112_ABCD_HCP_A_HCP_YA_Phenotype_Description_184.pt'), map_location=device)
    text_feature = text_feature[0:text_feature.shape[0], :]
    dataset = HCPfmriScoreDataset_sbjnum(path, name)
    csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/0112_ABCD_HCP_A_HCP_YA_reshape_Phenotype_Value_184.csv'))  #pheno_value
    traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/0112_ABCD_HCP_A_HCP_YA_Phenotype_Description_184.csv')) # description

    # csvdata_head = csvdata.columns.tolist()
    select_score = np.zeros((csvdata.shape[1] - 1, csvdata.shape[0]))

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

     # save dataset_sbj_fname as .csv



    ###score

    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]

    #
    for i in range(select_score_2.shape[0]):
        select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(
            ~np.isnan(select_score_2[i, :]))[0]] - np.mean(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])

    y_arr = select_score_2
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    # (select_score_2[6,:] - np.mean(select_score_2[6,:])) / np.std(select_score_2[6,:])    select_score_2[6,np.where(~np.isnan(select_score_2[6,:]))[0]]

    ###other
    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])

    dataset_x = np.reshape(dataset.data.x, (select_fname.size, 400, 400))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 400))

    dataset_pos = np.reshape(dataset.data.pos, (select_fname.size, 400, 400))
    dataset_pos = dataset_pos[dataset_indices[0], :, :]
    dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0] * dataset_pos.shape[1], 400))

    dataset_indices_torch = torch.from_numpy(dataset_indices[0])

    # att_indices = np.where(dataset_indices_torch == dataset.data.edge_sbj_torch[ :, np.newaxis])

    att_indices = np.arange(0, dataset.data.edge_sbj_torch[:, np.newaxis].shape[0])
    dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0], :]
    dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
    dataset_edge_attr = dataset.data.edge_attr[att_indices[0], :]

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

    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=dataset.data.y.size()[0], fold=fold)
    tr_index = np.concatenate((tr_index, te_index))
    # tr_index = np.concatenate((tr_index, te_index, val_index))

    # tr_index = tr_index[:-(tr_index.shape[0]%10)]
    # val_index = val_index[:-(val_index.shape[0]%10)]

    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[val_index]



    return train_dataset, val_dataset, test_dataset, text_feature #, dataset, train_index1, val_index1



def data_load_abcd_fmri(path, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    fold = opt.fold


    text_feature = torch.load(os.path.join(opt.phenoroot + '/1223_ABCD_Phenotype_Description_115.pt'), map_location=device)
    text_feature = text_feature[0:text_feature.shape[0], :]
    dataset = HCPfmriScoreDataset_sbjnum(path, name)
    csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/1223_ABCD_Phenotype_Value_115.csv'))  #pheno_value
    traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/1223_ABCD_Phenotype_Description_115.csv')) # description

    # csvdata_head = csvdata.columns.tolist()
    select_score = np.zeros((csvdata.shape[1] - 1, csvdata.shape[0]))

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

     # save dataset_sbj_fname as .csv



    ###score

    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]

    #
    for i in range(select_score_2.shape[0]):
        select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(
            ~np.isnan(select_score_2[i, :]))[0]] - np.mean(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])

    y_arr = select_score_2
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    # (select_score_2[6,:] - np.mean(select_score_2[6,:])) / np.std(select_score_2[6,:])    select_score_2[6,np.where(~np.isnan(select_score_2[6,:]))[0]]

    ###other
    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])

    dataset_x = np.reshape(dataset.data.x, (9438, 400, 400))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 400))

    dataset_pos = np.reshape(dataset.data.pos, (9438, 400, 400))
    dataset_pos = dataset_pos[dataset_indices[0], :, :]
    dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0] * dataset_pos.shape[1], 400))

    dataset_indices_torch = torch.from_numpy(dataset_indices[0])

    # att_indices = np.where(dataset_indices_torch == dataset.data.edge_sbj_torch[ :, np.newaxis])

    att_indices = np.arange(0, dataset.data.edge_sbj_torch[:, np.newaxis].shape[0])
    dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0], :]
    dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
    dataset_edge_attr = dataset.data.edge_attr[att_indices[0], :]

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

    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=dataset.data.y.size()[0], fold=fold)
    tr_index = np.concatenate((tr_index, te_index))

    # tr_index = tr_index[:-(tr_index.shape[0]%10)]
    # val_index = val_index[:-(val_index.shape[0]%10)]

    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[val_index]



    return train_dataset, val_dataset, test_dataset, text_feature #, dataset, train_index1, val_index1



def data_load_hcp_d_fmri(path, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    fold = opt.fold
    opt.phenoroot = '/data/hzb/project/BodyDecoding_data/HCP_D/HCP_D_Phenotype'

    text_feature = torch.load(os.path.join(opt.phenoroot + '/0109_HCP_D_Phenotype_Description_34.pt'), map_location=device)
    text_feature = text_feature[0:text_feature.shape[0], :]
    # text_feature = text_feature[0:72, :]

    dataset = HCPfmriScoreDataset_sbjnum(path, name)
    csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/0109_HCP_D_Phenotype_Value_34.csv'))  #pheno_value
    traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/0109_HCP_D_Phenotype_Description_34.csv')) # description

    # csvdata_head = csvdata.columns.tolist()
    select_score = np.zeros((csvdata.shape[1] - 1, csvdata.shape[0]))

    for i in range(traincsvdata['var_name'].shape[0]):
        select_score[i] = csvdata[traincsvdata['var_name'][i]]

    # non_nan_columns = np.where(~np.isnan(select_score).any(axis=0))[0]
    # select_score_1 = select_score[:, non_nan_columns]
    # csv_fname_values = csvdata["Subject"][non_nan_columns].values

    select_score_1 = select_score
    csv_fname_values = csvdata["sbj_name"].values

    dataset.data.sbj_fname = [(x) for x in dataset.data.sbj_fname]
    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)

     # save dataset_sbj_fname as .csv



    ###score

    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]

    #
    for i in range(select_score_2.shape[0]):
        select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(
            ~np.isnan(select_score_2[i, :]))[0]] - np.mean(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])

    y_arr = select_score_2
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    # (select_score_2[6,:] - np.mean(select_score_2[6,:])) / np.std(select_score_2[6,:])    select_score_2[6,np.where(~np.isnan(select_score_2[6,:]))[0]]

    ###other
    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])

    dataset_x = np.reshape(dataset.data.x, (620, 400, 400))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 400))

    dataset_pos = np.reshape(dataset.data.pos, (620, 400, 400))
    dataset_pos = dataset_pos[dataset_indices[0], :, :]
    dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0] * dataset_pos.shape[1], 400))

    dataset_indices_torch = torch.from_numpy(dataset_indices[0])

    # att_indices = np.where(dataset_indices_torch == dataset.data.edge_sbj_torch[ :, np.newaxis])

    att_indices = np.arange(0, dataset.data.edge_sbj_torch[:, np.newaxis].shape[0])
    dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0], :]
    dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
    dataset_edge_attr = dataset.data.edge_attr[att_indices[0], :]

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

    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=dataset.data.y.size()[0], fold=fold)
    tr_index = np.concatenate((tr_index, te_index))

    # tr_index = tr_index[:-(tr_index.shape[0]%10)]
    # val_index = val_index[:-(val_index.shape[0]%10)]

    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[val_index]



    return train_dataset, val_dataset, test_dataset, text_feature #, dataset, train_index1, val_index1



def data_load_hcp_fmri(path, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    fold = opt.fold

    opt.phenoroot = '/data/hzb/project/BodyDecoding_data/HCP/HCP_Phenotype'
    text_feature = torch.load(os.path.join(opt.phenoroot + '/0227_HCP_YA_Phenotype_Description_35_domain.pt'), map_location=device)  #0227_HCP_YA_Phenotype_Description_35_domain,   1223_HCP_YA_Phenotype_Description_35
    text_feature = text_feature[0:text_feature.shape[0], :]
    dataset = HCPfmriScoreDataset_sbjnum(path, name)
    csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/1223_HCP_YA_Phenotype_Value_35.csv'))  #pheno_value
    traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/1223_HCP_YA_Phenotype_Description_35.csv')) # description




########################abcd
    # text_feature = torch.load(os.path.join(opt.phenoroot + '/1223_ABCD_Phenotype_Description_115.pt'), map_location=device)
    # text_feature = text_feature[0:text_feature.shape[0], :]
    # dataset = HCPfmriScoreDataset_sbjnum(path, name)
    # csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/1223_ABCD_Phenotype_Value_115.csv'))  #pheno_value
    # traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/1223_ABCD_Phenotype_Description_115.csv')) # description



########################hcp-ya orig

    # opt.datadir = '/data/hzb/project/BodyDecoding/data'
    # opt.csvroot = '/data/hzb/project/BodyDecoding/data/HCP_pcp/HCP_Phenotype'
    # text_feature = torch.load(os.path.join(opt.datadir + '/HCP_phenotype_feature_109_1030.pt'), map_location=device)
    #
    # dataset = HCPfmriScoreDataset_sbjnum(path, name)
    # csvdata = pd.read_csv(os.path.join(opt.csvroot + '/S900_Release_Subjects_select_1030.csv'))
    # traincsvdata = pd.read_csv(os.path.join(opt.csvroot + '/S900_phenotype_description_select_1030.csv'))


####################



    # csvdata_head = csvdata.columns.tolist()
    select_score = np.zeros((csvdata.shape[1] - 1, csvdata.shape[0]))

    for i in range(traincsvdata['var_name'].shape[0]):
        select_score[i] = csvdata[traincsvdata['var_name'][i]]

    # non_nan_columns = np.where(~np.isnan(select_score).any(axis=0))[0]
    # select_score_1 = select_score[:, non_nan_columns]
    # csv_fname_values = csvdata["Subject"][non_nan_columns].values

    select_score_1 = select_score
    csv_fname_values = csvdata["Subject"].values

    dataset.data.sbj_fname = [int(x) for x in dataset.data.sbj_fname]
    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)

    ###score

    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]

    #
    for i in range(select_score_2.shape[0]):
        select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(
            ~np.isnan(select_score_2[i, :]))[0]] - np.mean(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])

    y_arr = select_score_2
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    # (select_score_2[6,:] - np.mean(select_score_2[6,:])) / np.std(select_score_2[6,:])    select_score_2[6,np.where(~np.isnan(select_score_2[6,:]))[0]]

    ###other
    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])

    dataset_x = np.reshape(dataset.data.x, (873, 400, 400))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 400))

    dataset_pos = np.reshape(dataset.data.pos, (873, 400, 400))
    dataset_pos = dataset_pos[dataset_indices[0], :, :]
    dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0] * dataset_pos.shape[1], 400))

    dataset_indices_torch = torch.from_numpy(dataset_indices[0])

    att_indices = np.where(dataset_indices_torch == dataset.data.edge_sbj_torch[:, np.newaxis])

    dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0], :]
    dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
    dataset_edge_attr = dataset.data.edge_attr[att_indices[0], :]

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

    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=dataset.data.y.size()[0], fold=fold)
    # tr_index = np.concatenate((tr_index, te_index, val_index))
    tr_index = np.concatenate((tr_index, te_index))

    # tr_index = tr_index[:-(tr_index.shape[0]%10)]
    # val_index = val_index[:-(val_index.shape[0]%10)]

    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[val_index]


    return train_dataset, val_dataset, test_dataset, text_feature #, dataset, train_index1, val_index1


def data_load_hcp_nan_test_fmri(path, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    fold = opt.fold

    opt.phenoroot = '/data/hzb/project/BodyDecoding/data/HCP_NAN_test_pcp/HCP_Phenotype'
    text_feature = torch.load(os.path.join(opt.phenoroot + '/1223_HCP_YA_Phenotype_Description_35.pt'), map_location=device)
    text_feature = text_feature[0:text_feature.shape[0], :]
    dataset = HCPfmriScoreDataset_sbjnum(path, name)
    csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/1223_HCP_NAN_test_Phenotype_Value_35.csv'))  #pheno_value
    traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/1223_HCP_YA_Phenotype_Description_35.csv')) # description






    # csvdata_head = csvdata.columns.tolist()
    select_score = np.zeros((csvdata.shape[1] - 1, csvdata.shape[0]))

    for i in range(traincsvdata['var_name'].shape[0]):
        select_score[i] = csvdata[traincsvdata['var_name'][i]]

    # non_nan_columns = np.where(~np.isnan(select_score).any(axis=0))[0]
    # select_score_1 = select_score[:, non_nan_columns]
    # csv_fname_values = csvdata["Subject"][non_nan_columns].values

    select_score_1 = select_score
    csv_fname_values = csvdata["Subject"].values

    dataset.data.sbj_fname = [int(x) for x in dataset.data.sbj_fname]
    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)

    ###score

    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]

    #
    for i in range(select_score_2.shape[0]):
        select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(
            ~np.isnan(select_score_2[i, :]))[0]] - np.mean(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])

    y_arr = select_score_2
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    # (select_score_2[6,:] - np.mean(select_score_2[6,:])) / np.std(select_score_2[6,:])    select_score_2[6,np.where(~np.isnan(select_score_2[6,:]))[0]]

    ###other
    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])

    dataset_x = np.reshape(dataset.data.x, (873, 400, 400))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 400))

    dataset_pos = np.reshape(dataset.data.pos, (873, 400, 400))
    dataset_pos = dataset_pos[dataset_indices[0], :, :]
    dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0] * dataset_pos.shape[1], 400))

    dataset_indices_torch = torch.from_numpy(dataset_indices[0])

    att_indices = np.where(dataset_indices_torch == dataset.data.edge_sbj_torch[:, np.newaxis])

    dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0], :]
    dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
    dataset_edge_attr = dataset.data.edge_attr[att_indices[0], :]

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

    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=dataset.data.y.size()[0], fold=fold)
    # tr_index = np.concatenate((tr_index, te_index, val_index))
    tr_index = np.concatenate((tr_index, te_index))

    # tr_index = tr_index[:-(tr_index.shape[0]%10)]
    # val_index = val_index[:-(val_index.shape[0]%10)]

    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[val_index]


    return train_dataset, val_dataset, test_dataset, text_feature #, dataset, train_index1, val_index1


def data_load_hcp_a_fmri(path, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    fold = opt.fold

    opt.phenoroot = '/data/hzb/project/BodyDecoding_data/HCP_A/HCP_A_Phenotype'
    text_feature = torch.load(os.path.join(opt.phenoroot + '/1223_HCP_A_Phenotype_Description_34.pt'), map_location=device)
    text_feature = text_feature[0:text_feature.shape[0], :]
    dataset = HCPfmriScoreDataset_sbjnum(path, name)
    csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/1223_HCP_A_Phenotype_Value_34.csv'))  #pheno_value
    traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/1223_HCP_A_Phenotype_Description_34.csv')) # description




    # csvdata_head = csvdata.columns.tolist()
    select_score = np.zeros((csvdata.shape[1] - 1, csvdata.shape[0]))

    for i in range(traincsvdata['var_name'].shape[0]):
        select_score[i] = csvdata[traincsvdata['var_name'][i]]

    # non_nan_columns = np.where(~np.isnan(select_score).any(axis=0))[0]
    # select_score_1 = select_score[:, non_nan_columns]
    # csv_fname_values = csvdata["Subject"][non_nan_columns].values

    select_score_1 = select_score
    csv_fname_values = csvdata["subject_id_fmri"].values

    dataset.data.sbj_fname = [(x) for x in dataset.data.sbj_fname]
    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)

    ###score

    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]

    #
    for i in range(select_score_2.shape[0]):
        select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(
            ~np.isnan(select_score_2[i, :]))[0]] - np.mean(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])

    y_arr = select_score_2
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    # (select_score_2[6,:] - np.mean(select_score_2[6,:])) / np.std(select_score_2[6,:])    select_score_2[6,np.where(~np.isnan(select_score_2[6,:]))[0]]

    ###other
    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])

    dataset_x = np.reshape(dataset.data.x, (y_arr.shape[0], 400, 400))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 400))

    dataset_pos = np.reshape(dataset.data.pos, (y_arr.shape[0], 400, 400))
    dataset_pos = dataset_pos[dataset_indices[0], :, :]
    dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0] * dataset_pos.shape[1], 400))

    dataset_indices_torch = torch.from_numpy(dataset_indices[0])

    att_indices = np.where(dataset_indices_torch == dataset.data.edge_sbj_torch[:, np.newaxis])

    dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0], :]
    dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
    dataset_edge_attr = dataset.data.edge_attr[att_indices[0], :]

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

    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=dataset.data.y.size()[0], fold=fold)
    # tr_index = np.concatenate((tr_index, te_index, val_index))
    tr_index = np.concatenate((tr_index, te_index))

    # tr_index = tr_index[:-(tr_index.shape[0]%10)]
    # val_index = val_index[:-(val_index.shape[0]%10)]

    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[val_index]


    return train_dataset, val_dataset, test_dataset, text_feature #, dataset, train_index1, val_index1


def data_load_hcp_a_intra_fmri(path, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    fold = opt.fold

    opt.phenoroot = '/data/hzb/project/BodyDecoding_data/HCP_A/HCP_A_Phenotype'
    text_feature = torch.load(os.path.join(opt.phenoroot + '/0110_HCP_A_train_Phenotype_Description_22.pt'), map_location=device)
    text_feature = text_feature[0:text_feature.shape[0], :]

    test_text_feature = torch.load(os.path.join(opt.phenoroot + '/0110_HCP_A_test_Phenotype_Description_12.pt'), map_location=device)
    text_text_feature = test_text_feature[0:test_text_feature.shape[0], :]


    dataset = HCPfmriScoreDataset_sbjnum(path, name)
    csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/1223_HCP_A_Phenotype_Value_34.csv'))  #pheno_value
    traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/0110_HCP_A_train_Phenotype_Description_22.csv')) # description
    testcsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/0110_HCP_A_test_Phenotype_Description_12.csv')) # description
    all_csvdata = pd.concat([traincsvdata['var_name'], testcsvdata['var_name']], ignore_index=True)



    # csvdata_head = csvdata.columns.tolist()
    select_score = np.zeros((csvdata.shape[1] - 1, csvdata.shape[0]))

    for i in range(all_csvdata.shape[0]):
        select_score[i] = csvdata[all_csvdata[i]]

    # non_nan_columns = np.where(~np.isnan(select_score).any(axis=0))[0]
    # select_score_1 = select_score[:, non_nan_columns]
    # csv_fname_values = csvdata["Subject"][non_nan_columns].values

    select_score_1 = select_score
    csv_fname_values = csvdata["subject_id_fmri"].values

    dataset.data.sbj_fname = [(x) for x in dataset.data.sbj_fname]
    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)

    ###score

    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]





    #
    for i in range(select_score_2.shape[0]):
        select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(
            ~np.isnan(select_score_2[i, :]))[0]] - np.mean(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])

    y_arr = select_score_2
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    # (select_score_2[6,:] - np.mean(select_score_2[6,:])) / np.std(select_score_2[6,:])    select_score_2[6,np.where(~np.isnan(select_score_2[6,:]))[0]]

    ###other
    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])

    dataset_x = np.reshape(dataset.data.x, (y_arr.shape[0], 400, 400))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 400))

    dataset_pos = np.reshape(dataset.data.pos, (y_arr.shape[0], 400, 400))
    dataset_pos = dataset_pos[dataset_indices[0], :, :]
    dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0] * dataset_pos.shape[1], 400))

    dataset_indices_torch = torch.from_numpy(dataset_indices[0])

    att_indices = np.where(dataset_indices_torch == dataset.data.edge_sbj_torch[:, np.newaxis])

    dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0], :]
    dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
    dataset_edge_attr = dataset.data.edge_attr[att_indices[0], :]

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

    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=dataset.data.y.size()[0], fold=fold)
    # tr_index = np.concatenate((tr_index, te_index, val_index))
    tr_index = np.concatenate((tr_index, te_index))

    # tr_index = tr_index[:-(tr_index.shape[0]%10)]
    # val_index = val_index[:-(val_index.shape[0]%10)]

    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[val_index]


    return train_dataset, val_dataset, test_dataset, text_feature, test_text_feature #, dataset, train_index1, val_index1


def data_load_hcp_d_intra_fmri(path, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    fold = opt.fold

    opt.phenoroot = '/data/hzb/project/BodyDecoding_data/HCP_D/HCP_D_Phenotype'
    text_feature = torch.load(os.path.join(opt.phenoroot + '/0110_HCP_D_train_Phenotype_Description_21.pt'), map_location=device)
    text_feature = text_feature[0:text_feature.shape[0], :]

    test_text_feature = torch.load(os.path.join(opt.phenoroot + '/0110_HCP_D_test_Phenotype_Description_13.pt'), map_location=device)
    text_text_feature = test_text_feature[0:test_text_feature.shape[0], :]


    dataset = HCPfmriScoreDataset_sbjnum(path, name)
    csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/0109_HCP_D_Phenotype_Value_34.csv'))  #pheno_value
    traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/0110_HCP_D_train_Phenotype_Description_21.csv')) # description
    testcsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/0110_HCP_D_test_Phenotype_Description_13.csv')) # description
    all_csvdata = pd.concat([traincsvdata['var_name'], testcsvdata['var_name']], ignore_index=True)



    # csvdata_head = csvdata.columns.tolist()
    select_score = np.zeros((csvdata.shape[1] - 1, csvdata.shape[0]))

    for i in range(all_csvdata.shape[0]):
        select_score[i] = csvdata[all_csvdata[i]]

    # non_nan_columns = np.where(~np.isnan(select_score).any(axis=0))[0]
    # select_score_1 = select_score[:, non_nan_columns]
    # csv_fname_values = csvdata["Subject"][non_nan_columns].values

    select_score_1 = select_score
    csv_fname_values = csvdata["sbj_name"].values

    dataset.data.sbj_fname = [(x) for x in dataset.data.sbj_fname]
    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)

    ###score

    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]





    #
    for i in range(select_score_2.shape[0]):
        select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(
            ~np.isnan(select_score_2[i, :]))[0]] - np.mean(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])

    y_arr = select_score_2
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    # (select_score_2[6,:] - np.mean(select_score_2[6,:])) / np.std(select_score_2[6,:])    select_score_2[6,np.where(~np.isnan(select_score_2[6,:]))[0]]

    ###other
    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])

    dataset_x = np.reshape(dataset.data.x, (y_arr.shape[0], 400, 400))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 400))

    dataset_pos = np.reshape(dataset.data.pos, (y_arr.shape[0], 400, 400))
    dataset_pos = dataset_pos[dataset_indices[0], :, :]
    dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0] * dataset_pos.shape[1], 400))

    dataset_indices_torch = torch.from_numpy(dataset_indices[0])

    att_indices = np.where(dataset_indices_torch == dataset.data.edge_sbj_torch[:, np.newaxis])

    dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0], :]
    dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
    dataset_edge_attr = dataset.data.edge_attr[att_indices[0], :]

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

    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=dataset.data.y.size()[0], fold=fold)
    # tr_index = np.concatenate((tr_index, te_index, val_index))
    tr_index = np.concatenate((tr_index, te_index))

    # tr_index = tr_index[:-(tr_index.shape[0]%10)]
    # val_index = val_index[:-(val_index.shape[0]%10)]

    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[val_index]


    return train_dataset, val_dataset, test_dataset, text_feature, test_text_feature #, dataset, train_index1, val_index1


def data_load_hcp_ya_intra_fmri(path, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    fold = opt.fold

    opt.phenoroot = '/data/hzb/project/BodyDecoding_data/HCP/HCP_Phenotype'
    text_feature = torch.load(os.path.join(opt.phenoroot + '/0110_HCP_YA_train_Phenotype_Description_23.pt'), map_location=device)
    text_feature = text_feature[0:text_feature.shape[0], :]

    test_text_feature = torch.load(os.path.join(opt.phenoroot + '/0110_HCP_YA_test_Phenotype_Description_12.pt'), map_location=device)
    text_text_feature = test_text_feature[0:test_text_feature.shape[0], :]


    dataset = HCPfmriScoreDataset_sbjnum(path, name)
    csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/1223_HCP_YA_Phenotype_Value_35.csv'))  #pheno_value
    traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/0110_HCP_YA_train_Phenotype_Description_23.csv')) # description
    testcsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/0110_HCP_YA_test_Phenotype_Description_12.csv')) # description
    all_csvdata = pd.concat([traincsvdata['var_name'], testcsvdata['var_name']], ignore_index=True)



    # csvdata_head = csvdata.columns.tolist()
    select_score = np.zeros((csvdata.shape[1] - 1, csvdata.shape[0]))

    for i in range(all_csvdata.shape[0]):
        select_score[i] = csvdata[all_csvdata[i]]

    # non_nan_columns = np.where(~np.isnan(select_score).any(axis=0))[0]
    # select_score_1 = select_score[:, non_nan_columns]
    # csv_fname_values = csvdata["Subject"][non_nan_columns].values

    select_score_1 = select_score
    csv_fname_values = csvdata["Subject"].values
    dataset.data.sbj_fname = [int(x) for x in dataset.data.sbj_fname]
    # dataset.data.sbj_fname = [(x) for x in dataset.data.sbj_fname]
    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)

    ###score

    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]





    #
    for i in range(select_score_2.shape[0]):
        select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(
            ~np.isnan(select_score_2[i, :]))[0]] - np.mean(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])

    y_arr = select_score_2
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    # (select_score_2[6,:] - np.mean(select_score_2[6,:])) / np.std(select_score_2[6,:])    select_score_2[6,np.where(~np.isnan(select_score_2[6,:]))[0]]

    ###other
    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])

    dataset_x = np.reshape(dataset.data.x, (y_arr.shape[0], 400, 400))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 400))

    dataset_pos = np.reshape(dataset.data.pos, (y_arr.shape[0], 400, 400))
    dataset_pos = dataset_pos[dataset_indices[0], :, :]
    dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0] * dataset_pos.shape[1], 400))

    dataset_indices_torch = torch.from_numpy(dataset_indices[0])

    att_indices = np.where(dataset_indices_torch == dataset.data.edge_sbj_torch[:, np.newaxis])

    dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0], :]
    dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
    dataset_edge_attr = dataset.data.edge_attr[att_indices[0], :]

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

    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=dataset.data.y.size()[0], fold=fold)
    # tr_index = np.concatenate((tr_index, te_index, val_index))
    tr_index = np.concatenate((tr_index, te_index))

    # tr_index = tr_index[:-(tr_index.shape[0]%10)]
    # val_index = val_index[:-(val_index.shape[0]%10)]

    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[val_index]


    return train_dataset, val_dataset, test_dataset, text_feature, test_text_feature #, dataset, train_index1, val_index1


def data_load_abcd_intra_fmri(path, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    fold = opt.fold

    opt.phenoroot = '/data/hzb/project/BodyDecoding_data/ABCD/ABCD_Phenotype'
    text_feature = torch.load(os.path.join(opt.phenoroot + '/0110_ABCD_train_Phenotype_Description_71.pt'), map_location=device)
    text_feature = text_feature[0:text_feature.shape[0], :]

    test_text_feature = torch.load(os.path.join(opt.phenoroot + '/0110_ABCD_test_Phenotype_Description_44.pt'), map_location=device)
    text_text_feature = test_text_feature[0:test_text_feature.shape[0], :]


    dataset = HCPfmriScoreDataset_sbjnum(path, name)
    csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/1223_ABCD_Phenotype_Value_115.csv'))  #pheno_value
    traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/0110_ABCD_train_Phenotype_Description_71.csv')) # description
    testcsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/0110_ABCD_test_Phenotype_Description_44.csv')) # description
    all_csvdata = pd.concat([traincsvdata['var_name'], testcsvdata['var_name']], ignore_index=True)



    # csvdata_head = csvdata.columns.tolist()
    select_score = np.zeros((csvdata.shape[1] - 1, csvdata.shape[0]))

    for i in range(all_csvdata.shape[0]):
        select_score[i] = csvdata[all_csvdata[i]]

    # non_nan_columns = np.where(~np.isnan(select_score).any(axis=0))[0]
    # select_score_1 = select_score[:, non_nan_columns]
    # csv_fname_values = csvdata["Subject"][non_nan_columns].values

    select_score_1 = select_score
    csv_fname_values = csvdata["NAME"].values

    dataset.data.sbj_fname = [(x) for x in dataset.data.sbj_fname]
    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)

    ###score

    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]





    #
    for i in range(select_score_2.shape[0]):
        select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(
            ~np.isnan(select_score_2[i, :]))[0]] - np.mean(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])

    y_arr = select_score_2
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    # (select_score_2[6,:] - np.mean(select_score_2[6,:])) / np.std(select_score_2[6,:])    select_score_2[6,np.where(~np.isnan(select_score_2[6,:]))[0]]

    ###other
    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])

    dataset_x = np.reshape(dataset.data.x, (y_arr.shape[0], 400, 400))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 400))

    dataset_pos = np.reshape(dataset.data.pos, (y_arr.shape[0], 400, 400))
    dataset_pos = dataset_pos[dataset_indices[0], :, :]
    dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0] * dataset_pos.shape[1], 400))





    dataset_indices_torch = torch.from_numpy(dataset_indices[0])

    # att_indices = np.where(dataset_indices_torch == dataset.data.edge_sbj_torch[ :, np.newaxis])

    att_indices = np.arange(0, dataset.data.edge_sbj_torch[:, np.newaxis].shape[0])
    dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0], :]
    dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
    dataset_edge_attr = dataset.data.edge_attr[att_indices[0], :]




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

    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=dataset.data.y.size()[0], fold=fold)
    # tr_index = np.concatenate((tr_index, te_index, val_index))
    tr_index = np.concatenate((tr_index, te_index))

    # tr_index = tr_index[:-(tr_index.shape[0]%10)]
    # val_index = val_index[:-(val_index.shape[0]%10)]

    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[val_index]


    return train_dataset, val_dataset, test_dataset, text_feature, test_text_feature #, dataset, train_index1, val_index1


def data_load_abcd_fmri_delete_high(path, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    fold = opt.fold


    text_feature = torch.load(os.path.join(opt.phenoroot + '/1223_ABCD_Phenotype_Description_115.pt'), map_location=device)
    text_feature = text_feature[0:text_feature.shape[0], :]
    dataset = HCPfmriScoreDataset_sbjnum(path, name)
    csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/1223_ABCD_Phenotype_Value_115.csv'))  #pheno_value
    traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/1223_ABCD_Phenotype_Description_115.csv')) # description

    if opt.test_dataset=='HCP_A':
        test_text_feature = torch.load(os.path.join('/data/hzb/project/BodyDecoding_data/HCP_A/HCP_A_Phenotype/1223_HCP_A_Phenotype_Description_34.pt'), map_location=device)
    if opt.test_dataset=='HCP_D':
        test_text_feature = torch.load(os.path.join('/data/hzb/project/BodyDecoding_data/HCP_D/HCP_D_Phenotype/0109_HCP_D_Phenotype_Description_34.pt'),map_location=device)
    if opt.test_dataset=='HCP_YA':
        test_text_feature = torch.load(os.path.join('/data/hzb/project/BodyDecoding_data/HCP/HCP_Phenotype/1223_HCP_YA_Phenotype_Description_35.pt'),map_location=device)

    correlation_matrix = np.corrcoef(text_feature.cpu().numpy(), test_text_feature.cpu().numpy())  # 计算相关性矩阵
    row_correlation  = correlation_matrix[:text_feature.shape[0], text_feature.shape[0]:]
    row_correlation = torch.from_numpy(row_correlation)
    row_correlation = row_correlation.to(device)
    row_correlation_max =row_correlation.max(dim=1).values


#########################
    # num_top_percent = int(50 * 0.01 * row_correlation_max.size(0))
    #
    # top_values, top_indices = torch.topk(row_correlation_max, num_top_percent)
    # row_correlation2 = row_correlation
    # mask = torch.ones(row_correlation2.size(0), dtype=torch.bool)
    # mask[top_indices] = False
    # new_tensor = row_correlation2[mask]
    # new_tensor.max(dim=0).values

    ############################
    num_top_percent = int(opt.delete_percent*0.01 * row_correlation_max.size(0))

    top_values, top_indices = torch.topk(row_correlation_max, num_top_percent)
    top_indices = top_indices+1

    csvdata.iloc[:, top_indices.cpu().numpy()] = np.nan

    # csvdata_head = csvdata.columns.tolist()
    select_score = np.zeros((csvdata.shape[1] - 1, csvdata.shape[0]))

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

     # save dataset_sbj_fname as .csv



    ###score

    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]

    #
    for i in range(select_score_2.shape[0]):
        select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(
            ~np.isnan(select_score_2[i, :]))[0]] - np.mean(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])

    y_arr = select_score_2
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    # (select_score_2[6,:] - np.mean(select_score_2[6,:])) / np.std(select_score_2[6,:])    select_score_2[6,np.where(~np.isnan(select_score_2[6,:]))[0]]

    ###other
    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])

    dataset_x = np.reshape(dataset.data.x, (9438, 400, 400))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 400))

    dataset_pos = np.reshape(dataset.data.pos, (9438, 400, 400))
    dataset_pos = dataset_pos[dataset_indices[0], :, :]
    dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0] * dataset_pos.shape[1], 400))

    dataset_indices_torch = torch.from_numpy(dataset_indices[0])

    # att_indices = np.where(dataset_indices_torch == dataset.data.edge_sbj_torch[ :, np.newaxis])

    att_indices = np.arange(0, dataset.data.edge_sbj_torch[:, np.newaxis].shape[0])
    dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0], :]
    dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
    dataset_edge_attr = dataset.data.edge_attr[att_indices[0], :]

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

    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=dataset.data.y.size()[0], fold=fold)
    tr_index = np.concatenate((tr_index, te_index))

    # tr_index = tr_index[:-(tr_index.shape[0]%10)]
    # val_index = val_index[:-(val_index.shape[0]%10)]

    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[val_index]



    return train_dataset, val_dataset, test_dataset, text_feature #, dataset, train_index1, val_index1
