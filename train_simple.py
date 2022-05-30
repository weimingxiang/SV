import utilities as ut
from pudb import set_trace
import pandas as pd
import random
import numpy as np
import torch
import os
from multiprocessing import Pool, cpu_count
import pysam
import torchvision



data_dir = "../datasets/NA12878_PacBio_MtSinai/"

bam_path = data_dir + "sorted_final_merged.bam"

vcf_filename = data_dir + "insert_result_data.csv.vcf"


# get chr list
sam_file = pysam.AlignmentFile(bam_path, "rb")
chr_list = sam_file.references
chr_length = sam_file.lengths
sam_file.close()

hight = 224

# all_p_img = torch.empty(22199, 3+4+4, hight, hight)
# all_n_img = torch.empty(22199, 3+3+4, hight, hight)

# all_p_list = torch.empty(22199, 512, 9)
# all_n_list = torch.empty(22199, 512, 9)

wait = ["chr3", "chr4", "chr5", "chr6", "chr12", "chr14", "chrX"]

index = 0

# pool = Pool(2)
for chromosome, chr_len in zip(chr_list, chr_length):
    print("======= deal " + chromosome + " =======")

    p_position = torch.load(data_dir + 'position/' + chromosome + '/positive' + '.pt')
    n_position = torch.load(data_dir + 'position/' + chromosome + '/negative' + '.pt')

    if chromosome not in wait:
        index += len(p_position)
        continue



    print("img start")
    save_path = data_dir + 'image/' + chromosome


    t_positive_img = torch.load(save_path + '/positive_img' + '.pt')
    t_negative_img = torch.load(save_path + '/negative_img' + '.pt')
    positive_img_mid = torch.load(save_path + '/positive_img_mids' + '.pt')
    negative_img_mid = torch.load(save_path + '/negative_img_mids' + '.pt')
    positive_cigar_img = torch.load(save_path + '/positive_cigar_new_img' + '.pt')
    negative_cigar_img = torch.load(save_path + '/negative_cigar_new_img' + '.pt')

    mid_sign_img = torch.load(data_dir + "chromosome_img/" + chromosome + "_m(i)d_sign12.pt")

    positive_img_i = torch.empty(len(p_position), 512, 11)
    negative_img_i = torch.empty(len(n_position), 512, 11)

    resize = torchvision.transforms.Resize([512, 11])

    for i, b_e in enumerate(p_position):
        positive_img_i[i] = resize(mid_sign_img[b_e[0]:b_e[1]].unsqueeze(0))
        print("===== finish(positive_img) " + chromosome + " " + str(i))


    for i, b_e in enumerate(n_position):
        negative_img_i[i] = resize(mid_sign_img[b_e[0]:b_e[1]].unsqueeze(0))

        print("===== finish(negative_img) " + chromosome + " " + str(i))


    # _positive_img, _negative_img = pool.starmap(ut.to_input_image, zip([positive_img, negative_img], [rd_depth_mean] * 2))


    # t_positive_img = torch.load(data_dir + 'image/' + chromosome + '/positive_img' + '.pt') # 3
    # # t_negative_img = torch.load(data_dir + 'image/' + chromosome + '/negative_img' + '.pt')
    # positive_img_mid = torch.load(data_dir + 'image/' + chromosome + '/positive_img_mids' + '.pt') # 3
    # # negative_img_mid = torch.load(data_dir + 'image/' + chromosome + '/negative_img_mid' + '.pt')
    # # positive_img_i = torch.load(data_dir + 'image/' + chromosome + '/positive_img_m(i)d' + '.pt')
    # # negative_img_i = torch.load(data_dir + 'image/' + chromosome + '/negative_img_m(i)d' + '.pt')
    # print("img end")
    # # img/positive_cigar_img
    # # print("cigar start")
    # positive_cigar_img = torch.load(data_dir + 'image/' + chromosome + '/positive_cigar_new_img' + '.pt') # 4
    # negative_cigar_img = torch.load(data_dir + 'image/' + chromosome + '/negative_cigar_new_img' + '.pt')
    # # positive_cigar_img = torch.cat([positive_cigar_img, positive_cigar_img6], 1)
    # # negative_cigar_img = torch.cat([negative_cigar_img, negative_cigar_img6], 1)
    # print("cigar end")

    ut.mymkdir(data_dir + '/positive_data')
    ut.mymkdir(data_dir + '/negative_data')
    for i in range(len(p_position)):
        torch.save([torch.cat([t_positive_img[i], positive_img_mid[i], positive_cigar_img[i]], 0), positive_img_i[i]], data_dir + '/positive_data/' + str(index) + '.pt')
    #     index += length

    # for i in range(len(n_position)):
        torch.save([torch.cat([t_negative_img[i], negative_img_mid[i], negative_cigar_img[i]], 0), negative_img_i[i]], data_dir + '/negative_data/' + str(index) + '.pt')
        index += 1

    # all_p_list[index:index + length] = positive_img_i
    # all_n_list[index:index + length] = negative_img_i

    # all_p_img[index:index + length, :3, :, :] = t_positive_img
    # # all_n_img[index:index + length, :3, :, :] = t_negative_img

    # # all_positive_img_mid[a2:a2 + length] = positive_img_mid
    # all_p_img[index:index + length, 3:7, :, :] = positive_img_mid
    # # all_negative_img_mid[b2:b2 + length] = negative_img_mid
    # # all_n_img[index:index + length, 3:6, :, :] = negative_img_mid


    # # all_positive_cigar_img[a3:a3 + length] = positive_cigar_img
    # all_p_img[index:index + length, 7:, :, :] = positive_cigar_img
    # all_negative_cigar_img[b3:b3 + length] = negative_cigar_img
    # all_n_img[index:index + length, 6:, :, :] = negative_cigar_img


    # all_positive_img_i_list[index:index + length] = positive_img_i_list
    # all_negative_img_i_list[index:index + length] = negative_img_i_list

    # index += length


# all_p_img = torch.cat([all_positive_img, all_positive_img_mid, all_positive_cigar_img], 1) # 3, 3, 3 + 6
# all_n_img = torch.cat([all_negative_img, all_negative_img_mid, all_negative_cigar_img], 1)

# all_p_img[:, :3, :, :] = all_positive_img
# all_p_img[:, 3:6, :, :] = all_positive_img_mid
# all_p_img[:, 6:, :, :] = all_positive_cigar_img

# all_n_img[:, :3, :, :] = all_negative_img
# all_n_img[:, 3:6, :, :] = all_negative_img_mid
# all_n_img[:, 6:, :, :] = all_negative_cigar_img

print("finish")

# torch.save(all_p_img, data_dir + '/all_p_img' + '.pt')
# # torch.save(all_n_img, data_dir + '/all_n_img' + '.pt')
# # torch.save(all_p_list, data_dir + '/all_p_list' + '.pt')
# # torch.save(all_n_list, data_dir + '/all_n_list' + '.pt')

# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# # all_p_img = torch.empty(22199, 3+3+4, hight, hight)
# all_n_img = all_p_img
# # all_n_img = torch.empty(22199, 3+3+4, hight, hight)

# # all_p_list = torch.empty(22199, 512, 9)
# # all_n_list = torch.empty(22199, 512, 9)

# index = 0

# # pool = Pool(2)
# for chromosome, chr_len in zip(chr_list, chr_length):
#     print("======= deal " + chromosome + " =======")
#     print("img start")
#     # t_positive_img = torch.load(data_dir + 'image/' + chromosome + '/positive_img' + '.pt') # 3
#     t_negative_img = torch.load(data_dir + 'image/' + chromosome + '/negative_img' + '.pt')
#     # positive_img_mid = torch.load(data_dir + 'image/' + chromosome + '/positive_img_mid' + '.pt') # 3
#     negative_img_mid = torch.load(data_dir + 'image/' + chromosome + '/negative_img_mids' + '.pt')
#     # positive_img_i = torch.load(data_dir + 'image/' + chromosome + '/positive_img_m(i)d' + '.pt')
#     # negative_img_i = torch.load(data_dir + 'image/' + chromosome + '/negative_img_m(i)d' + '.pt')
#     print("img end")
#     # img/positive_cigar_img
#     # print("cigar start")
#     # positive_cigar_img = torch.load(data_dir + 'image/' + chromosome + '/positive_cigar_new_img' + '.pt') # 4
#     negative_cigar_img = torch.load(data_dir + 'image/' + chromosome + '/negative_cigar_new_img' + '.pt')
#     # # positive_cigar_img = torch.cat([positive_cigar_img, positive_cigar_img6], 1)
#     # # negative_cigar_img = torch.cat([negative_cigar_img, negative_cigar_img6], 1)
#     # print("cigar end")

#     length = len(t_negative_img)

#     # all_p_list[index:index + length] = positive_img_i
#     # all_n_list[index:index + length] = negative_img_i

#     # all_p_img[index:index + length, :3, :, :] = t_positive_img
#     all_n_img[index:index + length, :3, :, :] = t_negative_img

#     # all_positive_img_mid[a2:a2 + length] = positive_img_mid
#     # all_p_img[index:index + length, 3:6, :, :] = positive_img_mid
#     # all_negative_img_mid[b2:b2 + length] = negative_img_mid
#     all_n_img[index:index + length, 3:7, :, :] = negative_img_mid


#     # all_positive_cigar_img[a3:a3 + length] = positive_cigar_img
#     # all_p_img[index:index + length, 6:, :, :] = positive_cigar_img
#     # all_negative_cigar_img[b3:b3 + length] = negative_cigar_img
#     all_n_img[index:index + length, 7:, :, :] = negative_cigar_img


#     # all_positive_img_i_list[index:index + length] = positive_img_i_list
#     # all_negative_img_i_list[index:index + length] = negative_img_i_list

#     index += length


# # all_p_img = torch.cat([all_positive_img, all_positive_img_mid, all_positive_cigar_img], 1) # 3, 3, 3 + 6
# # all_n_img = torch.cat([all_negative_img, all_negative_img_mid, all_negative_cigar_img], 1)

# # all_p_img[:, :3, :, :] = all_positive_img
# # all_p_img[:, 3:6, :, :] = all_positive_img_mid
# # all_p_img[:, 6:, :, :] = all_positive_cigar_img

# # all_n_img[:, :3, :, :] = all_negative_img
# # all_n_img[:, 3:6, :, :] = all_negative_img_mid
# # all_n_img[:, 6:, :, :] = all_negative_cigar_img

# print("finish")

# # torch.save(all_p_img, data_dir + '/all_p_img' + '.pt')
# torch.save(all_n_img, data_dir + '/all_n_img' + '.pt')
# # torch.save(all_p_list, data_dir + '/all_p_list' + '.pt')
# # torch.save(all_n_list, data_dir + '/all_n_list' + '.pt')