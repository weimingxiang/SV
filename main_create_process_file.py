import os
from datetime import datetime
# 打印时间函数
import subprocess
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

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
seed_everything(2022)


data_dir = "../datasets/NA12878_PacBio_MtSinai/"

bam_path = data_dir + "sorted_final_merged.bam"

vcf_filename = data_dir + "insert_result_data.csv.vcf"


sam_file = pysam.AlignmentFile(bam_path, "rb")
chr_list = sam_file.references
chr_length = sam_file.lengths
sam_file.close()

hight = 224

data_list = []
for chromosome, chr_len in zip(chr_list, chr_length):
    if not os.path.exists(data_dir + 'flag/' + chromosome + '.1txt'):
        data_list.append((chromosome, chr_len))


for chr, len in data_list:
    # subprocess.call("python create_process_file.py --chr " + chr + " --len " + str(len) + " > " + chr + ".txt 2&>1", shell=False)
    print("python create_process_file.py --chr " + chr + " --len " + str(len))
    # fd = open(chr + ".txt")
    subprocess.Popen("python create_process_file.py --chr " + chr + " --len " + str(len), shell=True)
