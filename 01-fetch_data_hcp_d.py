

from nilearn import datasets
import argparse
from imports import preprocess_data_hcp as Reader
import os
import shutil
import sys

# Input data variables     make .mat
code_folder = os.getcwd()
data_folder_orig = '/data/hzb/project/BodyDecoding_data/HCP_D/'
data_folder = '/data/hzb/project/BodyDecoding/data/HCP_D_pcp/Schaefer/filt_noglobal/'

if not os.path.exists(data_folder):
    os.makedirs(data_folder)
shutil.copyfile(os.path.join(data_folder_orig,'HCP_D_subjetc_id/Subject_ID.txt'), os.path.join(data_folder, 'subject_IDs.txt'))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(description='Download ABIDE data and compute functional connectivity matrices')
    parser.add_argument('--pipeline', default='cpac', type=str,
                        help='Pipeline to preprocess ABIDE data. Available options are ccs, cpac, dparsf and niak.'
                             ' default: cpac.')
    parser.add_argument('--atlas', default='Schaefer400',
                        help='Brain parcellation atlas. Options: ho, cc200 and cc400, default: cc200.')
    parser.add_argument('--download', default=False, type=str2bool,
                        help='Dowload data or just compute functional connectivity. default: True')
    args = parser.parse_args()
    print(args)

    params = dict()

    pipeline = args.pipeline
    atlas = args.atlas
    download = args.download

    # Files to fetch

    files = ['rois_' + atlas]





    subject_IDs = Reader.get_ids_hcp_d() #changed path to data path
    subject_IDs = subject_IDs.tolist()



    time_series = Reader.get_timeseries(subject_IDs, atlas)

    # Compute and save connectivity matrices
    Reader.subject_connectivity(time_series, subject_IDs, atlas, 'correlation')
    Reader.subject_connectivity(time_series, subject_IDs, atlas, 'partial correlation')


if __name__ == '__main__':
    main()
