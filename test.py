import utilities as ut
from pudb import set_trace
import pandas as pd
import random
import numpy as np
import torch
import os
from multiprocessing import Pool, cpu_count
import pysam

data_dir = "../datasets/NA12878_PacBio_MtSinai/"


# all_p_img = torch.load(data_dir + '/all_p_img13' + '.pt')
# all_p_img = all_p_img[:, :11]
# torch.save(all_p_img, data_dir + '/all_p_img' + '.pt')

print("all_p_img")

all_n_img = torch.load(data_dir + '/all_n_img13' + '.pt')
all_n_img = all_n_img[:, :11]
torch.save(all_n_img, data_dir + '/all_n_img' + '.pt')

print("all_n_img")