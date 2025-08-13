from nilearn import datasets
import argparse
from imports import preprocess_data as Reader
import os
import shutil
import sys
import numpy as np



# subject_id = np.loadtxt('/data/hzb/project/BodyDecoding_data/ABCD/ABCD_subject_id/Subject_ID.txt', delimiter=',')


with open('/data/hzb/project/BodyDecoding_data/ABCD/ABCD_subject_id/Subject_ID.txt', 'r', encoding = 'utf-8') as file:
        subject_id = file.readlines()

for i in range(len(subject_id)):
    folder_name = str((subject_id[i]))
    os.makedirs('data/ABCD_pcp/Schaefer/filt_noglobal/'+folder_name[:-1])  # makedirs 创建文件时如果路径不存在会创建这个路径





