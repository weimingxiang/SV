import utilities as ut
from pudb import set_trace
import pandas as pd
import random
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
import os
from net import IDENet
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from multiprocessing import Pool, cpu_count
import pysam
import time
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest import Repeater
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
TuneReportCheckpointCallback


seed_everything(2022)

data_dir = "../datasets/NA12878_PacBio_MtSinai/"

bam_path = data_dir + "sorted_final_merged.bam"

vcf_filename = data_dir + "insert_result_data.csv.vcf"

# get chr list
sam_file = pysam.AlignmentFile(bam_path, "rb")
chr_list = sam_file.references
chr_length = sam_file.lengths
sam_file.close()

hight = 224


all_positive_img = torch.empty(0, 3, hight, hight)
all_negative_img = torch.empty(0, 3, hight, hight)

all_positive_img_mid = torch.empty(0, 3, hight, hight)
all_negative_img_mid = torch.empty(0, 3, hight, hight)

all_positive_cigar_img = torch.empty(0, 7, hight, hight)
all_negative_cigar_img = torch.empty(0, 7, hight, hight)

all_p_list = torch.empty(0, 512, 9)
all_n_list = torch.empty(0, 512, 9)

# pool = Pool(2)
for chromosome, chr_len in zip(chr_list, chr_length):
    print("======= deal " + chromosome + " =======")

    print("position start")

    p_position = torch.load(data_dir + 'position/' + chromosome + '/positive' + '.pt')
    n_position = torch.load(data_dir + 'position/' + chromosome + '/negative' + '.pt')

    print("img start")

    # chromosome_sign
    chromosome_sign = torch.load(data_dir + "chromosome_sign/" + chromosome + ".pt")
    mid_sign = torch.load(data_dir + "chromosome_sign/" + chromosome + "_mids_sign.pt")
    mid_sign_list = torch.load(data_dir + "chromosome_sign/" + chromosome + "_m(i)d_sign.pt")
    mid_sign_img = torch.load(data_dir + "chromosome_img/" + chromosome + "_m(i)d_sign.pt")

    rd_depth_mean = torch.mean(chromosome_sign[2].float())

    positive_img = [[] for _ in range(len(p_position))]
    negative_img = [[] for _ in range(len(n_position))]
    positive_img_mid = torch.empty(len(p_position), 4, hight, hight)
    negative_img_mid = torch.empty(len(n_position), 4, hight, hight)
    positive_img_i = torch.empty(len(p_position), 512, 9)
    negative_img_i = torch.empty(len(n_position), 512, 9)

    resize = torchvision.transforms.Resize([512, 9])

    for i, b_e in enumerate(p_position):
        positive_img[i] = chromosome_sign[:, b_e[0]:b_e[1]] # dim 3
        positive_img_mid[i] = ut.to_img_mid_single(mid_sign[:, b_e[0]:b_e[1]]) # dim 3
        positive_img_i[i] = resize(mid_sign_img[b_e[0]:b_e[1]].unsqueeze(0))
        print("===== finish(positive_img) " + chromosome + " " + str(i))


    for i, b_e in enumerate(n_position):
        negative_img[i] = chromosome_sign[:, b_e[0]:b_e[1]]
        negative_img_mid[i] = ut.to_img_mid_single(mid_sign[:, b_e[0]:b_e[1]]) # dim 3
        negative_img_i[i] = resize(mid_sign_img[b_e[0]:b_e[1]].unsqueeze(0))

        print("===== finish(negative_img) " + chromosome + " " + str(i))


    # _positive_img, _negative_img = pool.starmap(ut.to_input_image, zip([positive_img, negative_img], [rd_depth_mean] * 2))
    t_positive_img = ut.to_input_image(positive_img, rd_depth_mean)
    t_negative_img = ut.to_input_image(negative_img, rd_depth_mean)
    print("save image start")

    save_path = data_dir + 'image/' + chromosome

    ut.mymkdir(save_path)
    # pool.starmap(torch.save, zip([_positive_img, _negative_img, positive_cigar_img, negative_cigar_img], [save_path + '/positive_img' + '.pt', save_path + '/negative_img' + '.pt', save_path + '/positive_cigar_img' + '.pt', save_path + '/negative_cigar_img' + '.pt']))
    torch.save(t_positive_img, save_path + '/positive_img' + '.pt')
    torch.save(t_negative_img, save_path + '/negative_img' + '.pt')
    torch.save(positive_img_mid, save_path + '/positive_img_mids' + '.pt')
    torch.save(negative_img_mid, save_path + '/negative_img_mids' + '.pt')
    torch.save(positive_img_i, save_path + '/positive_img_m(i)d' + '.pt')
    torch.save(negative_img_i, save_path + '/negative_img_m(i)d' + '.pt')
    print("img end")

    all_positive_img = torch.cat((all_positive_img, t_positive_img), 0)
    all_negative_img = torch.cat((all_negative_img, t_negative_img), 0)
    all_positive_cigar_img = torch.cat((all_positive_cigar_img, positive_cigar_img), 0)
    all_negative_cigar_img = torch.cat((all_negative_cigar_img, negative_cigar_img), 0)
    all_positive_img_mid = torch.cat((all_positive_img_mid, positive_img_mid), 0)
    all_negative_img_mid = torch.cat((all_negative_img_mid, negative_img_mid), 0)

    all_p_list = torch.cat((all_p_list, positive_img_i), 0)
    # set_trace()
    all_n_list = torch.cat((all_n_list, positive_img_i), 0)


all_p_img = torch.cat([all_positive_img, all_positive_img_mid, all_positive_cigar_img], 1) # 3, 3, 3
all_n_img = torch.cat([all_negative_img, all_negative_img_mid, all_negative_cigar_img], 1)

torch.save(all_p_img, data_dir + '/all_p_img' + '.pt')
torch.save(all_n_img, data_dir + '/all_n_img' + '.pt')
torch.save(all_p_list, data_dir + '/all_p_list' + '.pt')
torch.save(all_n_list, data_dir + '/all_n_list' + '.pt')