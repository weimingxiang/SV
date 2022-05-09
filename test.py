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

bam_path = data_dir + "sorted_final_merged.bam"

vcf_filename = data_dir + "insert_result_data.csv.vcf"


# get chr list
sam_file = pysam.AlignmentFile(bam_path, "rb")
chr_list = sam_file.references
chr_length = sam_file.lengths
sam_file.close()

hight = 224

all_p_img = torch.empty(22199, 3+3, hight, hight)
all_n_img = torch.empty(22199, 3+3, hight, hight)
all_p_img7 = torch.empty(22199, 7, hight, hight)
all_n_img7 = torch.empty(22199, 7, hight, hight)

index = 0

# pool = Pool(2)
for chromosome, chr_len in zip(chr_list, chr_length):
    print("======= deal " + chromosome + " =======")


    print("img start")
    t_positive_img = torch.load(data_dir + 'image/' + chromosome + '/positive_img' + '.pt') # 3
    t_negative_img = torch.load(data_dir + 'image/' + chromosome + '/negative_img' + '.pt')
    positive_img_mid = torch.load(data_dir + 'image/' + chromosome + '/positive_img_mid' + '.pt') # 3
    negative_img_mid = torch.load(data_dir + 'image/' + chromosome + '/negative_img_mid' + '.pt')
    print("img end")

    # img/positive_cigar_img
    print("cigar start")
    positive_cigar_img = torch.load(data_dir + 'image/' + chromosome + '/positive_cigar_img' + '.pt') # 3
    negative_cigar_img = torch.load(data_dir + 'image/' + chromosome + '/negative_cigar_img' + '.pt')
    positive_cigar_img4 = torch.load(data_dir + 'image6/' + chromosome + '/positive_cigar_img' + '.pt') # 4
    negative_cigar_img4 = torch.load(data_dir + 'image6/' + chromosome + '/negative_cigar_img' + '.pt')
    # positive_cigar_img = torch.cat([positive_cigar_img, positive_cigar_img6], 1)
    # negative_cigar_img = torch.cat([negative_cigar_img, negative_cigar_img6], 1)
    print("cigar end")

    length = len(t_positive_img)


    all_p_img[index:index + length, :3, :, :] = t_positive_img
    all_n_img[index:index + length, :3, :, :] = t_negative_img

    # all_positive_img_mid[a2:a2 + length] = positive_img_mid
    all_p_img[index:index + length, 3:, :, :] = positive_img_mid
    # all_negative_img_mid[b2:b2 + length] = negative_img_mid
    all_n_img[index:index + length, 3:, :, :] = negative_img_mid


    # all_positive_cigar_img[a3:a3 + length] = positive_cigar_img
    all_p_img7[index:index + length, :3, :, :] = positive_cigar_img
    # all_negative_cigar_img[b3:b3 + length] = negative_cigar_img
    all_n_img7[index:index + length, :3, :, :] = negative_cigar_img

    all_p_img7[index:index + length, 3:, :, :] = positive_cigar_img4
    all_n_img7[index:index + length, 3:, :, :] = negative_cigar_img4

    index += length


# all_p_img = torch.cat([all_positive_img, all_positive_img_mid, all_positive_cigar_img], 1) # 3, 3, 3 + 6
# all_n_img = torch.cat([all_negative_img, all_negative_img_mid, all_negative_cigar_img], 1)

# all_p_img[:, :3, :, :] = all_positive_img
# all_p_img[:, 3:6, :, :] = all_positive_img_mid
# all_p_img[:, 6:, :, :] = all_positive_cigar_img

# all_n_img[:, :3, :, :] = all_negative_img
# all_n_img[:, 3:6, :, :] = all_negative_img_mid
# all_n_img[:, 6:, :, :] = all_negative_cigar_img

print("finish")

torch.save(all_p_img, data_dir + '/all_p_img' + '.pt')
torch.save(all_n_img, data_dir + '/all_n_img' + '.pt')
torch.save(all_p_img7, data_dir + '/all_p_img7' + '.pt')
torch.save(all_n_img7, data_dir + '/all_n_img7' + '.pt')
