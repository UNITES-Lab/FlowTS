import os
import torch
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname('__file__'), '../'))
from Utils.context_fid import Context_FID
from Utils.metric_utils import display_scores
from Utils.cross_correlation import CrossCorrelLoss
import os

iterations = 5


### setting of metrics
fid = 1
corr = 1
dis = 1
pred = 1


data_name = os.getenv('data_name')
data_len = int(os.getenv('data_len'))
name = data_name

# import ipdb; ipdb.set_trace()
path_prefix = 'Checkpoints_' +  str(data_name) + '_' + str(data_len)


if data_len == 24:
    if name == 'energy':
        ori_data = np.load(path_prefix + '/samples/energy_norm_truth_24_train.npy')  # Uncomment the line if dataset other than Sine is used.
        fake_data = np.load(path_prefix + '/ddpm_fake_energy_24.npy')
    if name == 'etth':
        ori_data = np.load(path_prefix + '/samples/etth_norm_truth_24_train.npy')  # Uncomment the line if dataset other than Sine is used.
        fake_data = np.load(path_prefix + '/ddpm_fake_etth_24.npy')
    if name == 'fmri':
        ori_data = np.load(path_prefix + '/samples/fMRI_norm_truth_24_train.npy')  # Uncomment the line if dataset other than Sine is used.
        fake_data = np.load(path_prefix + '/ddpm_fake_fmri_24.npy')
    if name == 'mujoco':
        ori_data = np.load(path_prefix + '/samples/mujoco_norm_truth_24_train.npy')  # Uncomment the line if dataset other than Sine is used.
        fake_data = np.load(path_prefix + '/ddpm_fake_mujoco_24.npy')
    if name == 'sines':
        ori_data = np.load(path_prefix + '/samples/sine_ground_truth_24_train.npy')  # Uncomment the line if dataset other than Sine is used.
        fake_data = np.load(path_prefix + '/ddpm_fake_sines_24.npy')
    if name == 'stocks':
        ori_data = np.load(path_prefix + '/samples/stock_norm_truth_24_train.npy')  # Uncomment the line if dataset other than Sine is used.
        fake_data = np.load(path_prefix + '/ddpm_fake_stocks_24.npy')

if data_len == 64:
    if name == 'energy':
        ori_data = np.load(path_prefix + '/samples/energy_norm_truth_64_train.npy')  # Uncomment the line if dataset other than Sine is used.
        fake_data = np.load(path_prefix + '/ddpm_fake_energy_64.npy')
    if name == 'etth':
        ori_data = np.load(path_prefix + '/samples/etth_norm_truth_64_train.npy')  # Uncomment the line if dataset other than Sine is used.
        fake_data = np.load(path_prefix + '/ddpm_fake_etth_64.npy')


if data_len == 128:
    if name == 'energy':
        ori_data = np.load(path_prefix + '/samples/energy_norm_truth_128_train.npy')  # Uncomment the line if dataset other than Sine is used.
        fake_data = np.load(path_prefix + '/ddpm_fake_energy_128.npy')
    if name == 'etth':
        ori_data = np.load(path_prefix + '/samples/etth_norm_truth_128_train.npy')  # Uncomment the line if dataset other than Sine is used.
        fake_data = np.load(path_prefix + '/ddpm_fake_etth_128.npy')

if data_len == 256:
    if name == 'energy':
        ori_data = np.load(path_prefix + '/samples/energy_norm_truth_256_train.npy')  # Uncomment the line if dataset other than Sine is used.
        fake_data = np.load(path_prefix + '/ddpm_fake_energy_256.npy')
    if name == 'etth':
        ori_data = np.load(path_prefix + '/samples/etth_norm_truth_256_train.npy')  # Uncomment the line if dataset other than Sine is used.
        fake_data = np.load(path_prefix + '/ddpm_fake_etth_256.npy')

def random_choice(size, num_select=100):
    select_idx = np.random.randint(low=0, high=size, size=(num_select,))
    return select_idx

if fid:
    context_fid_score = []

    for i in range(iterations):
        context_fid = Context_FID(ori_data[:], fake_data[:ori_data.shape[0]])
        context_fid_score.append(context_fid)
        print(f'Iter {i}: ', 'context-fid =', context_fid, '\n')
        
    mean, sigma = display_scores(context_fid_score)
    content = f'fid {data_name} {data_len} {mean} {sigma}'

    with open('log.txt', 'a+') as file:
        file.write(content + '\n')






if corr:
    x_real = torch.from_numpy(ori_data)
    x_fake = torch.from_numpy(fake_data)
    correlational_score = []
    size = int(x_real.shape[0] / iterations)

    for i in range(iterations):
        real_idx = random_choice(x_real.shape[0], size)
        fake_idx = random_choice(x_fake.shape[0], size)
        corr = CrossCorrelLoss(x_real[real_idx, :, :], name='CrossCorrelLoss')
        loss = corr.compute(x_fake[fake_idx, :, :])
        correlational_score.append(loss.item())
        print(f'Iter {i}: ', 'cross-correlation =', loss.item(), '\n')

    mean, sigma = display_scores(correlational_score)
    content = f'corr {data_name} {data_len} {mean} {sigma}'

    with open('log.txt', 'a+') as file:
        file.write(content + '\n')


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append(os.path.join(os.path.dirname('__file__'), '../'))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from Utils.metric_utils import display_scores
from Utils.discriminative_metric import discriminative_score_metrics
from Utils.predictive_metric import predictive_score_metrics


if dis:


    discriminative_score = []

    # import ipdb; ipdb.set_trace()

    for i in range(iterations):
        temp_disc, fake_acc, real_acc = discriminative_score_metrics(ori_data[:], fake_data[:ori_data.shape[0]])
        discriminative_score.append(temp_disc)
        print(f'Iter {i}: ', temp_disc, ',', fake_acc, ',', real_acc, '\n')
        
    mean, sigma = display_scores(discriminative_score)
    content = f'disc {data_name} {data_len} {mean} {sigma}'

    with open('log.txt', 'a+') as file:
        file.write(content + '\n')

if pred:
    predictive_score = []
    for i in range(iterations):
        temp_pred = predictive_score_metrics(ori_data, fake_data[:ori_data.shape[0]])
        predictive_score.append(temp_pred)
        print(i, ' epoch: ', temp_pred, '\n')
        
    mean, sigma = display_scores(predictive_score)
    content = f'pred {data_name} {data_len} {mean} {sigma}'

    with open('log.txt', 'a+') as file:
        file.write(content + '\n')

