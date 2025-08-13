from nilearn import datasets
import argparse
from imports import preprocess_data as Reader
import os
import shutil
import sys
import numpy as np



subject_id = np.loadtxt('/data/hzb/project/BodyDecoding_data/HCP/HCP_subject_id/Subject_ID.txt', delimiter=',')
for i in range(len(subject_id)):
    folder_name = str(int(subject_id[i]))
    os.makedirs('data/HCP_pcp/Schaefer/filt_noglobal/'+folder_name)  # makedirs 创建文件时如果路径不存在会创建这个路径





