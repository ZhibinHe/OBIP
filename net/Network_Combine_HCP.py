import torch
import torch.nn as nn
from net.Network_Dual_ViT import *
from net.Network_Pred import *
from net.affinity_sink_layer import *
# from net.models_mae import *
import matplotlib.pyplot as plt
import pylab
import numpy as np




class CombinedModel_sink400(nn.Module):
    def __init__(self):
        super(CombinedModel_sink400, self).__init__()

        self.image_feature_extractor = Network_regress_score_out400(fmri_indim=400, fmri_outdim=1024, image_size=400,
                                                                 patch_size=400, num_classes=1024, dim=1024,
                                                                 depth=1, heads=16, mlp_dim=2048)

        self.phenotype_prediction = MultiTaskRegressionNetwork(num_tasks=315)
        self.affinity = Affinity(512)
        self.instNorm = nn.InstanceNorm2d(1, affine=True)

    def forward(self, fmri, text):

        features = self.image_feature_extractor(fmri)
        text1 = text.unsqueeze(0).expand(features.shape[0], text.shape[0], features.shape[2])

        s_s1 = self.affinity(features, text1)
        s_s1 = self.instNorm(s_s1[:, None, :, :]).squeeze(dim=1)
        log_s_s1 = sinkhorn_rpm(s_s1, n_iters=100, slack=False)
        s_s1 = torch.exp(log_s_s1)
        features_t = torch.transpose(features, 1, 2)
        features_mean = torch.matmul(features_t, s_s1)
        features1 = torch.transpose(features_mean, 1, 2)

        combine_feature = torch.cat((features1, text1), axis=2)

        prediction , regress_para= self.phenotype_prediction(combine_feature) #


        return prediction , regress_para



class CombinedModel_sink_feature400(nn.Module):
    def __init__(self):
        super(CombinedModel_sink_feature400, self).__init__()

        self.image_feature_extractor = Network_regress_score_out400(fmri_indim=400, fmri_outdim=1024, image_size=400,
                                                                 patch_size=400, num_classes=1024, dim=1024,
                                                                 depth=1, heads=16, mlp_dim=2048)

        self.phenotype_prediction = MultiTaskRegressionNetwork(num_tasks=35)     # ABCD 115    184    hcpa 34
        self.affinity = Affinity(512)
        self.instNorm = nn.InstanceNorm2d(1, affine=True)

    def forward(self, fmri, text):

        features = self.image_feature_extractor(fmri)
        text1 = text.unsqueeze(0).expand(features.shape[0], text.shape[0], features.shape[2])

        s_s1 = self.affinity(features, text1)
        s_s1 = self.instNorm(s_s1[:, None, :, :]).squeeze(dim=1)
        log_s_s1 = sinkhorn_rpm(s_s1, n_iters=100, slack=False)
        s_s1 = torch.exp(log_s_s1)

        features_t = torch.transpose(features, 1, 2)

        features_mean = torch.matmul(features_t, s_s1)
        features1 = torch.transpose(features_mean, 1, 2)

        combine_feature = torch.cat((features1, text1), axis=2)

        prediction , regress_para= self.phenotype_prediction(combine_feature) #
        return prediction , regress_para, combine_feature

        # import numpy as np
        # np.save('PAM_best_1127.npy', s_s1.cpu().numpy())
        # np.save('./log/analysis/HCPA_PAM_1.npy',s_s1.cpu().detach().numpy())

class CombinedModel_sink_feature400_numtask(nn.Module):
    def __init__(self, num_task):
        super(CombinedModel_sink_feature400_numtask, self).__init__()

        self.image_feature_extractor = Network_regress_score_out400(fmri_indim=400, fmri_outdim=1024, image_size=400,
                                                                 patch_size=400, num_classes=1024, dim=1024,
                                                                 depth=1, heads=16, mlp_dim=2048)

        self.phenotype_prediction = MultiTaskRegressionNetwork(num_tasks=num_task)     # ABCD 115    184
        self.affinity = Affinity(512)
        self.instNorm = nn.InstanceNorm2d(1, affine=True)

    def forward(self, fmri, text):

        features = self.image_feature_extractor(fmri)
        text1 = text.unsqueeze(0).expand(features.shape[0], text.shape[0], features.shape[2])

        s_s1 = self.affinity(features, text1)
        s_s1 = self.instNorm(s_s1[:, None, :, :]).squeeze(dim=1)
        log_s_s1 = sinkhorn_rpm(s_s1, n_iters=100, slack=False)
        s_s1 = torch.exp(log_s_s1)

        features_t = torch.transpose(features, 1, 2)

        features_mean = torch.matmul(features_t, s_s1)
        features1 = torch.transpose(features_mean, 1, 2)

        combine_feature = torch.cat((features1, text1), axis=2)

        prediction , regress_para= self.phenotype_prediction(combine_feature) #
        return prediction , regress_para, combine_feature

        # import numpy as np
        # np.save('PAM_best_1127.npy', s_s1.cpu().numpy())
