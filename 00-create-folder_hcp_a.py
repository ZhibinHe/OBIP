from nilearn import datasets
import argparse
from imports import preprocess_data as Reader
import os
import shutil
import sys
import numpy as np







with open('/data/hzb/project/BodyDecoding_data/HCP_A/HCP_A_subjetc_id/Subject_ID.txt', 'r', encoding='utf-8') as file:
        subject_id = file.readlines()

for i in range(len(subject_id)):
    folder_name = str((subject_id[i]))
    os.makedirs('data/HCP_A_pcp/Schaefer/filt_noglobal/' + folder_name[:-2])  # makedirs 创建文件时如果路径不存在会创建这个路径

