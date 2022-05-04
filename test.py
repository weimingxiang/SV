import utilities as ut
from pudb import set_trace
import pandas as pd
import random
import numpy as np
import torchvision
import math
import torch
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
import os
from net import IDENet
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from multiprocessing import Pool, cpu_count
import pysam
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest import Repeater
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback


os.environ["CUDA_VISIBLE_DEVICES"] = "3"


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
# pool = Pool(2)
for chromosome, chr_len in zip(chr_list, chr_length):
    print("deal " + chromosome)

    print("img start")
        # chromosome_sign

    chromosome_sign, mid_sign = ut.preprocess(bam_path, chromosome, chr_len, data_dir)
    torch.save(mid_sign, data_dir + "chromosome_sign/" + chromosome + "_mid_sign.pt")