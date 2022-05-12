
#from __future__ import print_function
import sys
import os
import random
from PIL import Image
import numpy as np
from pudb import set_trace
import argparse
from glob import glob
from bed2image import trans2img
import torch
import torchvision
from multiprocessing import Pool, cpu_count
import pysam
import time

hight = 224
resize = torchvision.transforms.Resize([hight, hight])

# # 通过pos计算值
# def pos2value(lis, pos):
#     pos_float = pos - int(pos)
#     x1 = lis[int(pos)] * (1 - pos_float)
#     x2 = lis[int(pos) + 1] * pos_float
#     return x1 + x2

# #计算上下四分位数
# #计算上下边缘
# #计算中位数
# def count_quartiles_median(lis):
#     length = float(len(lis)) - 1
#     q1 = length / 4
#     q3 = length * 3 / 4
#     q4 = q3 + 1.5 * (q3 - q1)
#     q = q1 - 1.5 * (q3 - q1)
#     q2 = length / 2  # 中位数
#     return pos2value(lis, q), pos2value(lis, q1), pos2value(lis, q2), pos2value(lis, q3), pos2value(lis, q4)

def get_rms(records):
    """
    均方根值 反映的是有效值而不是平均值
    """
    return np.sqrt(sum([x ** 2 for x in records]) / len(records))

# def get_gm(records):
#     """
#     几何平均
#     """
#     return (np.prod(records)) ** (1 / len(records))

# def get_hm(records):
#     """
#     调和平均
#     """
#     records = records
#     return len(records) / sum([1 / x for x in records])

def get_cv(records): #标准分和变异系数
    mean = np.mean(records)
    std = np.std(records)
    cv = std / mean
    return mean, std, cv

def mid_list2img(mid_sign_list, chromosome):
    mid_sign_img = torch.zeros(len(mid_sign_list), 9)
    for i, mid_sign in enumerate(mid_sign_list):
        if i % 50000 == 0:
            print(str(chromosome) + "\t" + str(i))
        mid_sign_img[i, 0] = len(mid_sign)
        if mid_sign_img[i, 0] == 1:
            continue
        mid_sign = np.array(mid_sign)
        mid_sign_img[i, 1], mid_sign_img[i, 2], mid_sign_img[i, 3], mid_sign_img[i, 4] = np.quantile(mid_sign, [0.25, 0.5, 0.75, 1], interpolation='linear')
        # mid_sign_img[i, 0], mid_sign_img[i, 1], mid_sign_img[i, 2], mid_sign_img[i, 3], mid_sign_img[i, 4] = count_quartiles_median(mid_sign) # 四分位
        # mid_sign_img[i, 5] = np.mean(mid_sign)
        # mid_sign_img[i, 6] = np.std(mid_sign)
        # mid_sign_img[i, 7] = len(mid_sign)
        mid_sign_img[i, 5] = get_rms(mid_sign)
        # mid_sign_img[i, 9] = get_gm(mid_sign)
        # mid_sign_img[i, 10] = get_hm(mid_sign)
        mid_sign_img[i, 6], mid_sign_img[i, 7], mid_sign_img[i, 8] = get_cv(mid_sign)

    return mid_sign_img



def mymkdir(mydir):
    if not os.path.exists(mydir):  #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(mydir)

def preprocess(bam_path, chromosome, chr_len, data_dir):

    return trans2img(bam_path, chromosome, chr_len, data_dir)

def to_input_image(imgs, rd_depth_mean, hight = 112):
    print("======= to input image begin =========")

    ims = torch.empty(len(imgs), 3, 224, 224)
    for i, img in enumerate(imgs):
        print("===== finish(to_input_image) " + str(i))
        img = img - rd_depth_mean + hight / 2
        img = torch.maximum(img, torch.tensor(0))

        pic_length = img.size()
        im = torch.zeros(3, pic_length[-1], hight)

        for x in range(pic_length[-1]):
            y = img[:, x].int()
            for j in range(pic_length[0]):
                im[j, x, :y[j]] = 255
                y_float = img[j, x] - y[j]
                im[j, x, y[j]:y[j]+1] = torch.round(255 * y_float)


        im = resize(im)
        ims[i] = im
    print("======= to input image end =========")
    return ims

class IdentifyDataset(torch.utils.data.Dataset):
    def __init__(self, positive_img, negative_img, p_list, n_list, p_index, n_index):

        self.positive_img = positive_img
        self.negative_img = negative_img
        self.p_list = p_list
        self.n_list = n_list
        self.p_index = p_index
        self.n_index = n_index

        # self._positive_img = to_input_image(positive_img, rd_depth_mean)
        # self._negative_img = to_input_image(negative_img, rd_depth_mean)
        self._len = len(positive_img) + len(negative_img)
        # print(self._len)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        if index % 2 ==0:
            return (self.positive_img[int(index / 2)], self.p_list[self.p_index[int(index / 2)]:self.p_index[int(index / 2) + 1]]), 1
        else:
            return (self.negative_img[int(index / 2)], self.n_list[self.n_index[int(index / 2)]:self.n_index[int(index / 2) + 1]]), 0


def MaxMinNormalization(x):
    Max = np.max(x)
    Min = np.min(x)
    x = (x - Min) / (Max - Min)
    return x

# def preprocess_cigar(bam_path, chromosome):
#     print("======= cigar image begin =========")
#     read_cigars = []
#     read_cigars_len = []
#     sam_file = pysam.AlignmentFile(bam_path, "rb")
#     chr_length = sam_file.lengths[sam_file.references.index(chromosome)]
#     refer_q_table = [[]] * chr_length

#     d0 = torch.tensor([[255], [0], [0], [0], [0], [0]])
#     d1 = torch.tensor([[0], [255], [0], [0], [0], [0]])
#     d2 = torch.tensor([[0], [0], [255], [0], [0], [0]])
#     d3 = torch.tensor([[0], [0], [0], [255], [0], [0]])
#     d4 = torch.tensor([[0], [0], [0], [0], [255], [0]])
#     d5 = torch.tensor([[0], [0], [0], [0], [0], [255]])

#     for index, read in enumerate(sam_file.fetch(chromosome)):
#         read_list = torch.empty(6, 0)
#         if refer_q_table[read.reference_start] == []:
#             refer_q_table[read.reference_start] = index


#         for operation, length in read.cigar: # (operation{6 class in 10 class}, length)
#             if operation == 0:
#                 read_list = torch.cat((read_list, d0.expand(6, length)), 1)
#             elif operation == 1:
#                 read_list = torch.cat((read_list, d1.expand(6, length)), 1)
#             elif operation == 2:
#                 read_list = torch.cat((read_list, d2.expand(6, length)), 1)
#             elif operation == 4:
#                 read_list = torch.cat((read_list, d3.expand(6, length)), 1)
#             elif operation == 5:
#                 read_list = torch.cat((read_list, d4.expand(6, length)), 1)
#             elif operation == 8:
#                 read_list = torch.cat((read_list, d5.expand(6, length)), 1)


#         read_cigars.append([read.reference_start, read_list])
#         read_cigars_len.append(read_list.size()[1] + read.reference_start)
#     sam_file.close()

#     for i in range(len(refer_q_table)):
#         if refer_q_table[i] == []:
#             if i == 0:
#                 refer_q_table[i] = 0
#             else:
#                 refer_q_table[i] = refer_q_table[i - 1]

#     print("======= cigar image end =========")
#     return read_cigars, read_cigars_len, refer_q_table


# def cigar_img(chromosome_cigar, chromosome_cigar_len, index_s, index_e):
#     # print("======= to input image begin =========")

#     if chromosome_cigar:
#         chromosome_cigar = chromosome_cigar[index_s, index_e]
#         chromosome_cigar_len = chromosome_cigar_len[index_s, index_e]


#         # mean = np.mean(read_length)
#         # std = np.std(read_length)
#         # maximum = mean + 3 * std # 提升图像信息量
#         maximum = np.max(chromosome_cigar_len)

#         cigars_img = torch.zeros([6, len(chromosome_cigar), int(maximum)])

#         for i, r_c in enumerate(chromosome_cigar):
#             read_list = r_c[1]
#             if chromosome_cigar_len[i] > maximum:
#                 cigars_img[:, i, r_c[0]:] = read_list[:, :maximum - r_c[0]]
#             else:
#                 cigars_img[:, i, r_c[0]:chromosome_cigar_len[i]] = read_list

#         cigars_img = resize(cigars_img)
#     else:
#         cigars_img = torch.zeros([6, hight, hight])

#     # print("======= to input image end =========")
#     return cigars_img



# def cigar_img_single(bam_path, chromosome, begin, end):
#     # print("======= cigar_img_single begin =========")
#     read_cigars = []
#     read_length = []
#     gap = "nan"
#     d = torch.tensor([[0], [0], [0], [0], [0], [0]])
#     d0 = torch.tensor([[255], [0], [0], [0], [0], [0]])
#     d1 = torch.tensor([[0], [255], [0], [0], [0], [0]])
#     d2 = torch.tensor([[0], [0], [255], [0], [0], [0]])
#     d3 = torch.tensor([[0], [0], [0], [255], [0], [0]])
#     d4 = torch.tensor([[0], [0], [0], [0], [255], [0]])
#     d5 = torch.tensor([[0], [0], [0], [0], [0], [255]])
#     sam_file = pysam.AlignmentFile(bam_path, "rb")

#     for read in sam_file.fetch(chromosome, begin, end):

#         if gap == "nan":
#             gap = read.reference_start - begin

#         read_list = torch.empty(6, 0) # 大小维护
#         empty = read.reference_start - begin
#         if gap >= 0:
#             read_list = torch.cat((read_list, d.expand(6, empty)), 1)
#         else:
#             read_list = torch.cat((read_list, d.expand(6, (empty - gap))), 1)

#         for operation, length in read.cigar: # (operation{10 class}, length)
#             if operation == 0:
#                 read_list = torch.cat((read_list, d0.expand(6, length)), 1)
#             elif operation == 1:
#                 read_list = torch.cat((read_list, d1.expand(6, length)), 1)
#             elif operation == 2:
#                 read_list = torch.cat((read_list, d2.expand(6, length)), 1)
#             elif operation == 4:
#                 read_list = torch.cat((read_list, d3.expand(6, length)), 1)
#             elif operation == 5:
#                 read_list = torch.cat((read_list, d4.expand(6, length)), 1)
#             elif operation == 8:
#                 read_list = torch.cat((read_list, d5.expand(6, length)), 1)

#         read_length.append(read_list.size()[1])
#         read_cigars.append(read_list)
#     sam_file.close()

#     if read_length:
#         # mean = np.mean(read_length)
#         # std = np.std(read_length)
#         # maximum = mean + 3 * std # 提升图像信息量
#         maximum = np.max(read_length)

#         cigars_img = torch.zeros([6, len(read_cigars), int(maximum)])

#         for i, r_c in enumerate(read_cigars):
#             if r_c.size()[1] > maximum:
#                 cigars_img[:, i, :] = r_c[:, :int(maximum)]
#             else:
#                 cigars_img[:, i, :r_c.size()[1]] = r_c

#         cigars_img = resize(cigars_img)
#     else:
#         cigars_img = torch.zeros([6, hight, hight])


#     # print("======= to input image end =========")
#     return cigars_img


def cigar_img_single_optimal(sam_file, chromosome, begin, end):
    # print("======= cigar_img_single begin =========")
    read_length = []
    gap = "nan"
    # sam_file = pysam.AlignmentFile(bam_path, "rb")

    for read in sam_file.fetch(chromosome, begin, end):

        if gap == "nan":
            gap = read.reference_start - begin

        read_list_terminal = 0
        empty = read.reference_start - begin
        if gap >= 0:
            read_list_terminal += empty
        else:
            read_list_terminal += empty - gap

        for operation, length in read.cigar: # (operation{10 class}, length)
            if operation == 0 or operation == 1 or operation == 2 or operation == 4 or operation == 5 or operation == 8:
                read_list_terminal += length

        read_length.append(read_list_terminal)


    if read_length:
        mean = np.mean(read_length)
        std = np.std(read_length)
        maximum = int(mean + 3 * std) # 提升图像信息量
        # maximum = np.max(read_length)

        cigars_img = torch.zeros([4, len(read_length), maximum])

        for i, read in enumerate(sam_file.fetch(chromosome, begin, end)):
            max_terminal = 0

            empty = read.reference_start - begin
            if gap >= 0:
                max_terminal = empty
            else:
                max_terminal = empty - gap

            for operation, length in read.cigar: # (operation{10 class}, length)
                if operation == 0:
                    if max_terminal+length < maximum:
                        cigars_img[0, i, max_terminal:max_terminal+length] = 255
                        max_terminal += length
                    else:
                        cigars_img[0, i, max_terminal:] = 255
                        break
                elif operation == 1:
                    if max_terminal+length < maximum:
                        cigars_img[1, i, max_terminal:max_terminal+length] = 255
                        max_terminal += length
                    else:
                        cigars_img[1, i, max_terminal:] = 255
                        break
                elif operation == 2:
                    if max_terminal+length < maximum:
                        cigars_img[2, i, max_terminal:max_terminal+length] = 255
                        max_terminal += length
                    else:
                        cigars_img[2, i, max_terminal:] = 255
                        break
                elif operation == 4:
                    if max_terminal+length < maximum:
                        cigars_img[3, i, max_terminal:max_terminal+length] = 255
                        max_terminal += length
                    else:
                        cigars_img[3, i, max_terminal:] = 255
                        break
                elif operation == 8:
                    if max_terminal+length < maximum:
                        max_terminal += length
                    else:
                        break
        cigars_img = resize(cigars_img)
    else:
        cigars_img = torch.zeros([4, hight, hight])

    # sam_file.close()
    # print("======= to input image end =========")
    return cigars_img



def cigar_new_img_single_optimal(sam_file, chromosome, begin, end): # 去除I的影响以对齐ref alignment
    # print("======= cigar_img_single begin =========")
    r_start = []
    r_end = []
    # sam_file = pysam.AlignmentFile(bam_path, "rb")

    for read in sam_file.fetch(chromosome, begin, end):
        r_start.append(read.reference_start)
        r_end.append(read.reference_end)


    if start:
        cigars_img = torch.zeros([4, len(start), np.max(r_end) - np.min(r_start)])

        for i, read in enumerate(sam_file.fetch(chromosome, begin, end)):
            max_terminal = 0

            empty = read.reference_start - begin
            if gap >= 0:
                max_terminal = empty
            else:
                max_terminal = empty - gap

            for operation, length in read.cigar: # (operation{10 class}, length)
                if operation == 0:
                    if max_terminal+length < maximum:
                        cigars_img[0, i, max_terminal:max_terminal+length] = 255
                        max_terminal += length
                    else:
                        cigars_img[0, i, max_terminal:] = 255
                        break
                elif operation == 1:
                    if max_terminal+length < maximum:
                        cigars_img[1, i, max_terminal:max_terminal+length] = 255
                        max_terminal += length
                    else:
                        cigars_img[1, i, max_terminal:] = 255
                        break
                elif operation == 2:
                    if max_terminal+length < maximum:
                        cigars_img[2, i, max_terminal:max_terminal+length] = 255
                        max_terminal += length
                    else:
                        cigars_img[2, i, max_terminal:] = 255
                        break
                elif operation == 4:
                    if max_terminal+length < maximum:
                        cigars_img[3, i, max_terminal:max_terminal+length] = 255
                        max_terminal += length
                    else:
                        cigars_img[3, i, max_terminal:] = 255
                        break
                elif operation == 8:
                    if max_terminal+length < maximum:
                        max_terminal += length
                    else:
                        break
        cigars_img = resize(cigars_img)
    else:
        cigars_img = torch.zeros([4, hight, hight])

    # sam_file.close()
    # print("======= to input image end =========")
    return cigars_img


# def cigar_img_single_optimal(sam_file, chromosome, begin, end):
#     # print("======= cigar_img_single begin =========")
#     read_length = []
#     gap = "nan"
#     # sam_file = pysam.AlignmentFile(bam_path, "rb")

#     for read in sam_file.fetch(chromosome, begin, end):

#         if gap == "nan":
#             gap = read.reference_start - begin

#         read_list_terminal = 0
#         empty = read.reference_start - begin
#         if gap >= 0:
#             read_list_terminal += empty
#         else:
#             read_list_terminal += empty - gap

#         for operation, length in read.cigar: # (operation{10 class}, length)
#             if operation == 0 or operation == 1 or operation == 2 or operation == 8:
#                 read_list_terminal += length

#         read_length.append(read_list_terminal)


#     if read_length:
#         mean = np.mean(read_length)
#         std = np.std(read_length)
#         maximum = int(mean + 3 * std) # 提升图像信息量
#         # maximum = np.max(read_length)

#         cigars_img = torch.zeros([3, len(read_length), maximum])

#         for i, read in enumerate(sam_file.fetch(chromosome, begin, end)):
#             max_terminal = 0

#             empty = read.reference_start - begin
#             if gap >= 0:
#                 max_terminal = empty
#             else:
#                 max_terminal = empty - gap

#             for operation, length in read.cigar: # (operation{10 class}, length)
#                 if operation == 0 or operation == 8:
#                     if max_terminal+length < maximum:
#                         cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 1:
#                     if max_terminal+length < maximum:
#                         cigars_img[1, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[1, i, max_terminal:] = 255
#                         break
#                 elif operation == 2:
#                     if max_terminal+length < maximum:
#                         cigars_img[2, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[2, i, max_terminal:] = 255
#                         break

#         cigars_img = resize(cigars_img)
#     else:
#         cigars_img = torch.zeros([3, hight, hight])

#     # sam_file.close()
#     # print("======= to input image end =========")
#     return cigars_img


# def cigar_img_single_optimal_time2sapce(sam_file, chromosome, begin, end):
#     # print("======= cigar_img_single begin =========")
#     read_length = []
#     gap = "nan"
#     # sam_file = pysam.AlignmentFile(bam_path, "rb")

#     for read in sam_file.fetch(chromosome, begin, end):

#         if gap == "nan":
#             gap = read.reference_start - begin

#         read_list_terminal = 0
#         empty = read.reference_start - begin
#         if gap >= 0:
#             read_list_terminal += empty
#         else:
#             read_list_terminal += empty - gap

#         for operation, length in read.cigar: # (operation{10 class}, length)
#             if operation == 0 or operation == 1 or operation == 2 or operation == 8:
#                 read_list_terminal += length

#         read_length.append(read_list_terminal)


#     if read_length:
#         mean = np.mean(read_length)
#         std = np.std(read_length)
#         maximum = int(mean + 3 * std) # 提升图像信息量
#         # maximum = np.max(read_length)

#         cigars_img = torch.zeros([1, len(read_length), maximum])

#         for i, read in enumerate(sam_file.fetch(chromosome, begin, end)):
#             max_terminal = 0

#             empty = read.reference_start - begin
#             if gap >= 0:
#                 max_terminal = empty
#             else:
#                 max_terminal = empty - gap

#             for operation, length in read.cigar: # (operation{10 class}, length)
#                 if operation == 0 or operation == 8:
#                     if max_terminal+length < maximum:
#                         cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 1:
#                     if max_terminal+length < maximum:
#                         # cigars_img[1, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[1, i, max_terminal:] = 255
#                         break
#                 elif operation == 2:
#                     if max_terminal+length < maximum:
#                         # cigars_img[2, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[2, i, max_terminal:] = 255
#                         break

#         cigars_img1 = resize(cigars_img)

#         cigars_img[:, :, :] = 0
#         for i, read in enumerate(sam_file.fetch(chromosome, begin, end)):
#             max_terminal = 0

#             empty = read.reference_start - begin
#             if gap >= 0:
#                 max_terminal = empty
#             else:
#                 max_terminal = empty - gap

#             for operation, length in read.cigar: # (operation{10 class}, length)
#                 if operation == 0 or operation == 8:
#                     if max_terminal+length < maximum:
#                         # cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 1:
#                     if max_terminal+length < maximum:
#                         cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 2:
#                     if max_terminal+length < maximum:
#                         # cigars_img[2, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[2, i, max_terminal:] = 255
#                         break

#         cigars_img2 = resize(cigars_img)

#         cigars_img[:, :, :] = 0
#         for i, read in enumerate(sam_file.fetch(chromosome, begin, end)):
#             max_terminal = 0

#             empty = read.reference_start - begin
#             if gap >= 0:
#                 max_terminal = empty
#             else:
#                 max_terminal = empty - gap

#             for operation, length in read.cigar: # (operation{10 class}, length)
#                 if operation == 0 or operation == 8:
#                     if max_terminal+length < maximum:
#                         # cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 1:
#                     if max_terminal+length < maximum:
#                         # cigars_img[1, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[1, i, max_terminal:] = 255
#                         break
#                 elif operation == 2:
#                     if max_terminal+length < maximum:
#                         cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[0, i, max_terminal:] = 255
#                         break

#         cigars_img3 = resize(cigars_img)

#         cigars_img = torch.cat([cigars_img1, cigars_img2, cigars_img3], dim = 0)
#     else:
#         cigars_img = torch.zeros([3, hight, hight])

#     # sam_file.close()
#     # print("======= to input image end =========")
#     return cigars_img

# # 6 dim
# def cigar_img_single_optimal(sam_file, chromosome, begin, end):
#     # print("======= cigar_img_single begin =========")
#     read_length = []
#     gap = "nan"
#     # sam_file = pysam.AlignmentFile(bam_path, "rb")

#     for read in sam_file.fetch(chromosome, begin, end):

#         if gap == "nan":
#             gap = read.reference_start - begin

#         read_list_terminal = 0
#         empty = read.reference_start - begin
#         if gap >= 0:
#             read_list_terminal += empty
#         else:
#             read_list_terminal += empty - gap

#         for operation, length in read.cigar: # (operation{10 class}, length)
#             if operation == 0 or operation == 1 or operation == 2 or operation == 4 or operation == 5 or operation == 8:
#                 read_list_terminal += length

#         read_length.append(read_list_terminal)


#     if read_length:
#         mean = np.mean(read_length)
#         std = np.std(read_length)
#         maximum = int(mean + 3 * std) # 提升图像信息量
#         # maximum = np.max(read_length)

#         cigars_img = torch.zeros([6, len(read_length), maximum])

#         for i, read in enumerate(sam_file.fetch(chromosome, begin, end)):
#             max_terminal = 0

#             empty = read.reference_start - begin
#             if gap >= 0:
#                 max_terminal = empty
#             else:
#                 max_terminal = empty - gap

#             for operation, length in read.cigar: # (operation{10 class}, length)
#                 if operation == 0:
#                     if max_terminal+length < maximum:
#                         cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 1:
#                     if max_terminal+length < maximum:
#                         cigars_img[1, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[1, i, max_terminal:] = 255
#                         break
#                 elif operation == 2:
#                     if max_terminal+length < maximum:
#                         cigars_img[2, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[2, i, max_terminal:] = 255
#                         break
#                 elif operation == 4:
#                     if max_terminal+length < maximum:
#                         cigars_img[3, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[3, i, max_terminal:] = 255
#                         break
#                 elif operation == 5:
#                     if max_terminal+length < maximum:
#                         cigars_img[4, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[4, i, max_terminal:] = 255
#                         break
#                 elif operation == 8:
#                     if max_terminal+length < maximum:
#                         cigars_img[5, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[5, i, max_terminal:] = 255
#                         break

#         cigars_img = resize(cigars_img)
#     else:
#         cigars_img = torch.zeros([6, hight, hight])

#     # sam_file.close()
#     # print("======= to input image end =========")
#     return cigars_img


# def cigar_img_single_optimal_time2sapce(sam_file, chromosome, begin, end):
#     # print("======= cigar_img_single begin =========")
#     read_length = []
#     gap = "nan"
#     # sam_file = pysam.AlignmentFile(bam_path, "rb")

#     for read in sam_file.fetch(chromosome, begin, end):

#         if gap == "nan":
#             gap = read.reference_start - begin

#         read_list_terminal = 0
#         empty = read.reference_start - begin
#         if gap >= 0:
#             read_list_terminal += empty
#         else:
#             read_list_terminal += empty - gap

#         for operation, length in read.cigar: # (operation{10 class}, length)
#             if operation == 0 or operation == 1 or operation == 2 or operation == 4 or operation == 5 or operation == 8:
#                 read_list_terminal += length

#         read_length.append(read_list_terminal)


#     if read_length:
#         mean = np.mean(read_length)
#         std = np.std(read_length)
#         maximum = int(mean + 3 * std) # 提升图像信息量
#         # maximum = np.max(read_length)

#         cigars_img = torch.zeros([3, len(read_length), maximum])

#         for i, read in enumerate(sam_file.fetch(chromosome, begin, end)):
#             max_terminal = 0

#             empty = read.reference_start - begin
#             if gap >= 0:
#                 max_terminal = empty
#             else:
#                 max_terminal = empty - gap

#             for operation, length in read.cigar: # (operation{10 class}, length)
#                 if operation == 0:
#                     if max_terminal+length < maximum:
#                         cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 1:
#                     if max_terminal+length < maximum:
#                         cigars_img[1, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[1, i, max_terminal:] = 255
#                         break
#                 elif operation == 2:
#                     if max_terminal+length < maximum:
#                         cigars_img[2, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[2, i, max_terminal:] = 255
#                         break
#                 elif operation == 4:
#                     if max_terminal+length < maximum:
#                         # cigars_img[3, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[3, i, max_terminal:] = 255
#                         break
#                 elif operation == 5:
#                     if max_terminal+length < maximum:
#                         # cigars_img[4, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[4, i, max_terminal:] = 255
#                         break
#                 elif operation == 8:
#                     if max_terminal+length < maximum:
#                         # cigars_img[5, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[5, i, max_terminal:] = 255
#                         break

#         cigars_img1 = resize(cigars_img)

#         cigars_img[:, :, :] = 0
#         for i, read in enumerate(sam_file.fetch(chromosome, begin, end)):
#             max_terminal = 0

#             empty = read.reference_start - begin
#             if gap >= 0:
#                 max_terminal = empty
#             else:
#                 max_terminal = empty - gap

#             for operation, length in read.cigar: # (operation{10 class}, length)
#                 if operation == 0:
#                     if max_terminal+length < maximum:
#                         # cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 1:
#                     if max_terminal+length < maximum:
#                         # cigars_img[1, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[1, i, max_terminal:] = 255
#                         break
#                 elif operation == 2:
#                     if max_terminal+length < maximum:
#                         # cigars_img[2, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[2, i, max_terminal:] = 255
#                         break
#                 elif operation == 4:
#                     if max_terminal+length < maximum:
#                         cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 5:
#                     if max_terminal+length < maximum:
#                         cigars_img[1, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[1, i, max_terminal:] = 255
#                         break
#                 elif operation == 8:
#                     if max_terminal+length < maximum:
#                         cigars_img[2, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[2, i, max_terminal:] = 255
#                         break

#         cigars_img2 = resize(cigars_img)

#         cigars_img = torch.cat([cigars_img1, cigars_img2], dim = 0)
#     else:
#         cigars_img = torch.zeros([6, hight, hight])

#     # sam_file.close()
#     # print("======= to input image end =========")
#     return cigars_img


# def cigar_img_single_optimal_time3sapce(sam_file, chromosome, begin, end):
#     # print("======= cigar_img_single begin =========")
#     read_length = []
#     gap = "nan"
#     # sam_file = pysam.AlignmentFile(bam_path, "rb")

#     for read in sam_file.fetch(chromosome, begin, end):

#         if gap == "nan":
#             gap = read.reference_start - begin

#         read_list_terminal = 0
#         empty = read.reference_start - begin
#         if gap >= 0:
#             read_list_terminal += empty
#         else:
#             read_list_terminal += empty - gap

#         for operation, length in read.cigar: # (operation{10 class}, length)
#             if operation == 0 or operation == 1 or operation == 2 or operation == 4 or operation == 5 or operation == 8:
#                 read_list_terminal += length

#         read_length.append(read_list_terminal)


#     if read_length:
#         mean = np.mean(read_length)
#         std = np.std(read_length)
#         maximum = int(mean + 3 * std) # 提升图像信息量
#         # maximum = np.max(read_length)

#         cigars_img = torch.zeros([2, len(read_length), maximum])

#         for i, read in enumerate(sam_file.fetch(chromosome, begin, end)):
#             max_terminal = 0

#             empty = read.reference_start - begin
#             if gap >= 0:
#                 max_terminal = empty
#             else:
#                 max_terminal = empty - gap

#             for operation, length in read.cigar: # (operation{10 class}, length)
#                 if operation == 0:
#                     if max_terminal+length < maximum:
#                         cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 1:
#                     if max_terminal+length < maximum:
#                         cigars_img[1, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[1, i, max_terminal:] = 255
#                         break
#                 elif operation == 2:
#                     if max_terminal+length < maximum:
#                         # cigars_img[2, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[2, i, max_terminal:] = 255
#                         break
#                 elif operation == 4:
#                     if max_terminal+length < maximum:
#                         # cigars_img[3, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[3, i, max_terminal:] = 255
#                         break
#                 elif operation == 5:
#                     if max_terminal+length < maximum:
#                         # cigars_img[4, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[4, i, max_terminal:] = 255
#                         break
#                 elif operation == 8:
#                     if max_terminal+length < maximum:
#                         # cigars_img[5, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[5, i, max_terminal:] = 255
#                         break

#         cigars_img1 = resize(cigars_img)

#         cigars_img[:, :, :] = 0
#         for i, read in enumerate(sam_file.fetch(chromosome, begin, end)):
#             max_terminal = 0

#             empty = read.reference_start - begin
#             if gap >= 0:
#                 max_terminal = empty
#             else:
#                 max_terminal = empty - gap

#             for operation, length in read.cigar: # (operation{10 class}, length)
#                 if operation == 0:
#                     if max_terminal+length < maximum:
#                         # cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 1:
#                     if max_terminal+length < maximum:
#                         # cigars_img[1, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[1, i, max_terminal:] = 255
#                         break
#                 elif operation == 2:
#                     if max_terminal+length < maximum:
#                         cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 4:
#                     if max_terminal+length < maximum:
#                         cigars_img[1, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[1, i, max_terminal:] = 255
#                         break
#                 elif operation == 5:
#                     if max_terminal+length < maximum:
#                         # cigars_img[1, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[1, i, max_terminal:] = 255
#                         break
#                 elif operation == 8:
#                     if max_terminal+length < maximum:
#                         # cigars_img[2, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[2, i, max_terminal:] = 255
#                         break

#         cigars_img2 = resize(cigars_img)

#         cigars_img[:, :, :] = 0
#         for i, read in enumerate(sam_file.fetch(chromosome, begin, end)):
#             max_terminal = 0

#             empty = read.reference_start - begin
#             if gap >= 0:
#                 max_terminal = empty
#             else:
#                 max_terminal = empty - gap

#             for operation, length in read.cigar: # (operation{10 class}, length)
#                 if operation == 0:
#                     if max_terminal+length < maximum:
#                         # cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 1:
#                     if max_terminal+length < maximum:
#                         # cigars_img[1, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[1, i, max_terminal:] = 255
#                         break
#                 elif operation == 2:
#                     if max_terminal+length < maximum:
#                         # cigars_img[2, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[2, i, max_terminal:] = 255
#                         break
#                 elif operation == 4:
#                     if max_terminal+length < maximum:
#                         # cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 5:
#                     if max_terminal+length < maximum:
#                         cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 8:
#                     if max_terminal+length < maximum:
#                         cigars_img[1, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[1, i, max_terminal:] = 255
#                         break

#         cigars_img3 = resize(cigars_img)

#         cigars_img = torch.cat([cigars_img1, cigars_img2, cigars_img3], dim = 0)
#     else:
#         cigars_img = torch.zeros([6, hight, hight])

#     # sam_file.close()
#     # print("======= to input image end =========")
#     return cigars_img


# def cigar_img_single_optimal_time6sapce(sam_file, chromosome, begin, end):
#     # print("======= cigar_img_single begin =========")
#     read_length = []
#     gap = "nan"
#     # sam_file = pysam.AlignmentFile(bam_path, "rb")

#     for read in sam_file.fetch(chromosome, begin, end):

#         if gap == "nan":
#             gap = read.reference_start - begin

#         read_list_terminal = 0
#         empty = read.reference_start - begin
#         if gap >= 0:
#             read_list_terminal += empty
#         else:
#             read_list_terminal += empty - gap

#         for operation, length in read.cigar: # (operation{10 class}, length)
#             if operation == 0 or operation == 1 or operation == 2 or operation == 4 or operation == 5 or operation == 8:
#                 read_list_terminal += length

#         read_length.append(read_list_terminal)


#     if read_length:
#         mean = np.mean(read_length)
#         std = np.std(read_length)
#         maximum = int(mean + 3 * std) # 提升图像信息量
#         # maximum = np.max(read_length)

#         cigars_img = torch.zeros([1, len(read_length), maximum])

#         for i, read in enumerate(sam_file.fetch(chromosome, begin, end)):
#             max_terminal = 0

#             empty = read.reference_start - begin
#             if gap >= 0:
#                 max_terminal = empty
#             else:
#                 max_terminal = empty - gap

#             for operation, length in read.cigar: # (operation{10 class}, length)
#                 if operation == 0:
#                     if max_terminal+length < maximum:
#                         cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 1:
#                     if max_terminal+length < maximum:
#                         # cigars_img[1, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[1, i, max_terminal:] = 255
#                         break
#                 elif operation == 2:
#                     if max_terminal+length < maximum:
#                         # cigars_img[2, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[2, i, max_terminal:] = 255
#                         break
#                 elif operation == 4:
#                     if max_terminal+length < maximum:
#                         # cigars_img[3, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[3, i, max_terminal:] = 255
#                         break
#                 elif operation == 5:
#                     if max_terminal+length < maximum:
#                         # cigars_img[4, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[4, i, max_terminal:] = 255
#                         break
#                 elif operation == 8:
#                     if max_terminal+length < maximum:
#                         # cigars_img[5, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[5, i, max_terminal:] = 255
#                         break

#         cigars_img1 = resize(cigars_img)

#         cigars_img[:, :, :] = 0
#         for i, read in enumerate(sam_file.fetch(chromosome, begin, end)):
#             max_terminal = 0

#             empty = read.reference_start - begin
#             if gap >= 0:
#                 max_terminal = empty
#             else:
#                 max_terminal = empty - gap

#             for operation, length in read.cigar: # (operation{10 class}, length)
#                 if operation == 0:
#                     if max_terminal+length < maximum:
#                         # cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 1:
#                     if max_terminal+length < maximum:
#                         cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 2:
#                     if max_terminal+length < maximum:
#                         # cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 4:
#                     if max_terminal+length < maximum:
#                         # cigars_img[1, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[1, i, max_terminal:] = 255
#                         break
#                 elif operation == 5:
#                     if max_terminal+length < maximum:
#                         # cigars_img[1, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[1, i, max_terminal:] = 255
#                         break
#                 elif operation == 8:
#                     if max_terminal+length < maximum:
#                         # cigars_img[2, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[2, i, max_terminal:] = 255
#                         break

#         cigars_img2 = resize(cigars_img)

#         cigars_img[:, :, :] = 0
#         for i, read in enumerate(sam_file.fetch(chromosome, begin, end)):
#             max_terminal = 0

#             empty = read.reference_start - begin
#             if gap >= 0:
#                 max_terminal = empty
#             else:
#                 max_terminal = empty - gap

#             for operation, length in read.cigar: # (operation{10 class}, length)
#                 if operation == 0:
#                     if max_terminal+length < maximum:
#                         # cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 1:
#                     if max_terminal+length < maximum:
#                         # cigars_img[1, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[1, i, max_terminal:] = 255
#                         break
#                 elif operation == 2:
#                     if max_terminal+length < maximum:
#                         cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 4:
#                     if max_terminal+length < maximum:
#                         # cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 5:
#                     if max_terminal+length < maximum:
#                         # cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 8:
#                     if max_terminal+length < maximum:
#                         # cigars_img[1, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[1, i, max_terminal:] = 255
#                         break

#         cigars_img3 = resize(cigars_img)

#         cigars_img[:, :, :] = 0
#         for i, read in enumerate(sam_file.fetch(chromosome, begin, end)):
#             max_terminal = 0

#             empty = read.reference_start - begin
#             if gap >= 0:
#                 max_terminal = empty
#             else:
#                 max_terminal = empty - gap

#             for operation, length in read.cigar: # (operation{10 class}, length)
#                 if operation == 0:
#                     if max_terminal+length < maximum:
#                         # cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 1:
#                     if max_terminal+length < maximum:
#                         # cigars_img[1, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[1, i, max_terminal:] = 255
#                         break
#                 elif operation == 2:
#                     if max_terminal+length < maximum:
#                         # cigars_img[2, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[2, i, max_terminal:] = 255
#                         break
#                 elif operation == 4:
#                     if max_terminal+length < maximum:
#                         cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 5:
#                     if max_terminal+length < maximum:
#                         # cigars_img[4, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[4, i, max_terminal:] = 255
#                         break
#                 elif operation == 8:
#                     if max_terminal+length < maximum:
#                         # cigars_img[5, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[5, i, max_terminal:] = 255
#                         break

#         cigars_img4 = resize(cigars_img)

#         cigars_img[:, :, :] = 0
#         for i, read in enumerate(sam_file.fetch(chromosome, begin, end)):
#             max_terminal = 0

#             empty = read.reference_start - begin
#             if gap >= 0:
#                 max_terminal = empty
#             else:
#                 max_terminal = empty - gap

#             for operation, length in read.cigar: # (operation{10 class}, length)
#                 if operation == 0:
#                     if max_terminal+length < maximum:
#                         # cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 1:
#                     if max_terminal+length < maximum:
#                         # cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 2:
#                     if max_terminal+length < maximum:
#                         # cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 4:
#                     if max_terminal+length < maximum:
#                         # cigars_img[1, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[1, i, max_terminal:] = 255
#                         break
#                 elif operation == 5:
#                     if max_terminal+length < maximum:
#                         cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 8:
#                     if max_terminal+length < maximum:
#                         # cigars_img[2, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[2, i, max_terminal:] = 255
#                         break

#         cigars_img5 = resize(cigars_img)

#         cigars_img[:, :, :] = 0
#         for i, read in enumerate(sam_file.fetch(chromosome, begin, end)):
#             max_terminal = 0

#             empty = read.reference_start - begin
#             if gap >= 0:
#                 max_terminal = empty
#             else:
#                 max_terminal = empty - gap

#             for operation, length in read.cigar: # (operation{10 class}, length)
#                 if operation == 0:
#                     if max_terminal+length < maximum:
#                         # cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 1:
#                     if max_terminal+length < maximum:
#                         # cigars_img[1, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[1, i, max_terminal:] = 255
#                         break
#                 elif operation == 2:
#                     if max_terminal+length < maximum:
#                         # cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 4:
#                     if max_terminal+length < maximum:
#                         # cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 5:
#                     if max_terminal+length < maximum:
#                         # cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         # cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 8:
#                     if max_terminal+length < maximum:
#                         cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[0, i, max_terminal:] = 255
#                         break

#         cigars_img6 = resize(cigars_img)

#         cigars_img = torch.cat([cigars_img1, cigars_img2, cigars_img3, cigars_img4, cigars_img5, cigars_img6], dim = 0)
#     else:
#         cigars_img = torch.zeros([6, hight, hight])

#     # sam_file.close()
#     # print("======= to input image end =========")
#     return cigars_img


# def cigar_MID(sam_file, chromosome):
#     # print("======= cigar_img_single begin =========")
#     read_length = []
#     gap = "nan"
#     # sam_file = pysam.AlignmentFile(bam_path, "rb")

#     for read in sam_file.fetch(chromosome, begin, end):

#         if gap == "nan":
#             gap = read.reference_start - begin

#         read_list_terminal = 0
#         empty = read.reference_start - begin
#         if gap >= 0:
#             read_list_terminal += empty
#         else:
#             read_list_terminal += empty - gap

#         for operation, length in read.cigar: # (operation{10 class}, length)
#             if operation == 0 or operation == 1 or operation == 8:
#                 read_list_terminal += length

#         read_length.append(read_list_terminal)


#     if read_length:
#         # mean = np.mean(read_length)
#         # std = np.std(read_length)
#         # maximum = int(mean + 3 * std) # 提升图像信息量
#         maximum = np.max(read_length)

#         cigars_img = torch.zeros([3, len(read_length), maximum])

#         for i, read in enumerate(sam_file.fetch(chromosome, begin, end)):
#             max_terminal = 0

#             empty = read.reference_start - begin
#             if gap >= 0:
#                 max_terminal = empty
#             else:
#                 max_terminal = empty - gap

#             for operation, length in read.cigar: # (operation{10 class}, length)
#                 if operation == 0 or operation == 8:
#                     if max_terminal+length < maximum:
#                         cigars_img[0, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[0, i, max_terminal:] = 255
#                         break
#                 elif operation == 1:
#                     if max_terminal+length < maximum:
#                         cigars_img[1, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[1, i, max_terminal:] = 255
#                         break
#                 elif operation == 2:
#                     if max_terminal+length < maximum:
#                         cigars_img[2, i, max_terminal:max_terminal+length] = 255
#                         max_terminal += length
#                     else:
#                         cigars_img[2, i, max_terminal:] = 255
#                         break

#         cigars_img = resize(cigars_img)
#     else:
#         cigars_img = torch.zeros([3, hight, hight])

#     # sam_file.close()
#     # print("======= to input image end =========")
#     return cigars_img



def to_img_mid_single(img, hight = 224):
    # print("======= to id image begin =========")

    img = torch.maximum(img.float(), torch.tensor(0))

    pic_length = img.size()
    im = torch.zeros(3, pic_length[-1], hight)

    for x in range(pic_length[-1]):
        y = img[:, x].int()
        for j in range(pic_length[0]):
            im[j, x, :y[j]] = 255
            y_float = img[j, x] - y[j]
            im[j, x, y[j]:y[j]+1] = torch.round(255 * y_float)

    im = resize(im)

    # print("======= to id image end =========")
    return im