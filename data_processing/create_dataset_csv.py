"""
Creation of dataset *.csv files for the fastmri dataset.

Copyright (c) 2019 Kerstin Hammernik <k.hammernik at imperial dot ac dot uk>
Department of Computing, Imperial College London, London, United Kingdom

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pandas as pd
import h5py
import xmltodict
import pprint
import argparse
import pathlib
import logging
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
     '--data-path', type=pathlib.Path, required=True,
     help='Path to the dataset',
)
parser.add_argument(
     '--csv-path', type=pathlib.Path, required=True,
     help='Path to the csv files',
)
parser.add_argument(
     '--dataset', type=str, required=True,
     help='Dataset for which the csv file should be generated.'
)

args = parser.parse_args()

image_type = '.h5'

print(f'Create dataset file for {args.dataset}')
output_name = f'{args.dataset}.csv'

# generate the file names
image_names = sorted([os.path.join(args.dataset, f) for f in os.listdir(os.path.join(args.data_path, args.dataset)) if f.endswith(image_type)])

# init dicts for infos that will be extracted from the dataset
img_info = {'filename' : image_names, 'type' : []}
acq_info = {'systemVendor' : [], 'systemModel' : [], 'systemFieldStrength_T' : [], 'receiverChannels' : [], 'institutionName' : [] }
seq_info = {'TR' : [] , 'TE' : [], 'TI': [], 'flipAngle_deg': [], 'sequence_type': [], 'echo_spacing': []}
enc_info = {'enc_x' : [], 'enc_y' : [], 'enc_z' : [], \
          'rec_x' : [], 'rec_y' : [], 'rec_z' : [], \
          'enc_x_mm' : [], 'enc_y_mm' : [], 'enc_z_mm' : [],
          'rec_x_mm' : [], 'rec_y_mm' : [], 'rec_z_mm' : [],
          'nPE' : []}
acc_info = {'acc' : [], 'num_low_freq' : []}

for fname in tqdm(image_names):
     dset =  h5py.File(os.path.join(args.data_path, fname),'r')
     img_info['type'].append(dset.attrs['acquisition'])
     acc_info['acc'].append(dset.attrs['acceleration'] if 'acceleration' in dset.attrs.keys() else 0)
     acc_info['num_low_freq'].append(dset.attrs['num_low_frequency'] if 'num_low_frequency' in dset.attrs.keys() else 0)
     header_xml = dset['ismrmrd_header'][()]
     header = xmltodict.parse(header_xml)['ismrmrdHeader']
     #pprint.pprint(header)   
     for key in acq_info.keys():
          acq_info[key].append(header['acquisitionSystemInformation'][key])
     for key in seq_info.keys():
          if key in header['sequenceParameters']:
               seq_info[key].append(header['sequenceParameters'][key])
          else:
               seq_info[key].append('n/a')
     enc_info['nPE'].append(int(header['encoding']['encodingLimits']['kspace_encoding_step_1']['maximum'])+1)
     if int(header['encoding']['encodingLimits']['kspace_encoding_step_1']['minimum']) != 0:
          raise ValueError('be careful!')
     for diridx in ['x', 'y', 'z']:
          enc_info[f'enc_{diridx}'].append(header['encoding']['encodedSpace']['matrixSize'][diridx])
          enc_info[f'rec_{diridx}'].append(header['encoding']['reconSpace']['matrixSize'][diridx])
          enc_info[f'enc_{diridx}_mm'].append(header['encoding']['encodedSpace']['fieldOfView_mm'][diridx])
          enc_info[f'rec_{diridx}_mm'].append(header['encoding']['reconSpace']['fieldOfView_mm'][diridx])

data_info = {**img_info, **acq_info, **enc_info, **acc_info, **seq_info}

# convert to pandas
df = pd.DataFrame(data_info)
print(df)

# save to output
print(f'Save csv file to {os.path.join(args.csv_path, output_name)}')
args.csv_path.mkdir(parents=True, exist_ok=True)
df.to_csv(args.csv_path / output_name, index=False)
