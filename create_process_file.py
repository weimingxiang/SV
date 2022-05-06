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

position_enforcement_refresh = 0
img_enforcement_refresh = 0
sign_enforcement_refresh = 0 # attention
cigar_enforcement_refresh = 0

# sam_file = pysam.AlignmentFile(bam_path, "rb")
# chr_list = sam_file.references
# chr_length = sam_file.lengths
# sam_file.close()

hight = 224

# data_list = []
# for chromosome, chr_len in zip(chr_list, chr_length):
#     if not os.path.exists(data_dir + 'flag/' + chromosome + '.txt'):
#         data_list.append((chromosome, chr_len))

def process(bam_path, chromosome, pic_length, data_dir):

    ref_chromosome_filename = data_dir + "chr/" + chromosome + ".fa"
    # fa = pysam.FastaFile(ref_chromosome_filename)
    # chr_string = fa.fetch(chromosome)
    sam_file = pysam.AlignmentFile(bam_path, "rb")

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    conjugate_m = torch.zeros(pic_length, dtype=torch.int)
    conjugate_i = torch.zeros(pic_length, dtype=torch.int)
    conjugate_d = torch.zeros(pic_length, dtype=torch.int)
    # match_count = torch.zeros(pic_length, dtype=torch.int)
    # mismatch_count = torch.zeros(pic_length, dtype=torch.int)
    # bam_op_count = torch.zeros([9, pic_length], dtype=torch.int)

    for read in sam_file.fetch(chromosome):
        if read.is_unmapped:
            continue
        start, end = (read.reference_start, read.reference_end)
        if start % 100 == 0:
            print(str(chromosome) + " " + str(start))

        # ref_read = chr_string[start:end]

        # read = read.get_forward_sequence()

        reference_index = start # % 2 == 0 :1  % 2 == 1 :0
        for operation, length in read.cigar: # (operation, length)
            if operation == 3 or operation == 7 or operation == 8:
                reference_index += length
            elif operation == 0:
                conjugate_m[reference_index:reference_index + length] += 1
                reference_index += length
            elif operation == 1:
                conjugate_i[reference_index] += length
            elif operation == 2:
                conjugate_d[reference_index:reference_index + length] += 1
                reference_index += length

    sam_file.close()

    # rd_count = MaxMinNormalization(rd_count)  # The scope of rd_count value is [0, 1]

    return torch.cat([conjugate_m.unsqueeze(0), conjugate_i.unsqueeze(0), conjugate_d.unsqueeze(0)], 0)

def p(sum_data):
    chromosome, chr_len = sum_data

    # copy begin
    print("deal " + chromosome)

    p_position = torch.load(data_dir + 'position/' + chromosome + '/positive' + '.pt')
    n_position = torch.load(data_dir + 'position/' + chromosome + '/negative' + '.pt')

    mid_sign = torch.load(data_dir + "chromosome_sign/" + chromosome + "_mid_sign.pt")
    # mid_sign = torch.load(data_dir + "chromosome_sign/" + chromosome + "_id_sign.pt")
    positive_img_id = torch.empty(len(p_position), 3, hight, hight)
    negative_img_id = torch.empty(len(n_position), 3, hight, hight)

    for i, b_e in enumerate(p_position):
        positive_img_id[i] = ut.to_img_id_single(mid_sign[:, b_e[0]:b_e[1]]) # dim 2
        print("===== finish(positive_img_id) " + chromosome + " " + str(i))


    for i, b_e in enumerate(n_position):
        negative_img_id[i] = ut.to_img_id_single(mid_sign[:, b_e[0]:b_e[1]]) # dim 2
        print("===== finish(negative_img_id) " + chromosome + " " + str(i))
    save_path = data_dir + 'image/' + chromosome
    torch.save(positive_img_id, save_path + '/positive_img_mid' + '.pt')
    torch.save(negative_img_id, save_path + '/negative_img_mid' + '.pt')

    # copy end
    torch.save(1, data_dir + 'flag/' + chromosome + '.1txt')





# pool = Pool()                # 创建进程池对象，进程数与multiprocessing.cpu_count()相同
# pool.imap_unordered(p, data_list)
# # pool.map(p, data_list)
# pool.close()
# pool.join()

# # p("chr1")


import argparse   #步骤一

def parse_args():
    """
    :return:进行参数的解析
    """
    description = "you should add those parameter"                   # 步骤二
    parser = argparse.ArgumentParser(description=description)        # 这些参数都有默认值，当调用parser.print_help()或者运行程序时由于参数不正确(此时python解释器其实也是调用了pring_help()方法)时，
                                                                     # 会打印这些描述信息，一般只需要传递description参数，如上。
    help = "The path of address"
    parser.add_argument('--chr',help = help)                   # 步骤三，后面的help是我的描述
    parser.add_argument('--len',help = help)                   # 步骤三，后面的help是我的描述
    args = parser.parse_args()                                       # 步骤四
    return args

if __name__ == '__main__':
    args = parse_args()
    # print(args.chr)            #直接这么获取即可。
    # print(type(args.chr))
    p([args.chr, int(args.len)])