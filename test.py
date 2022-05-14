import torch
from multiprocessing import Pool, cpu_count

data_dir = "../datasets/NA12878_PacBio_MtSinai/"

pool = Pool(2)

all_p_img, all_n_img = pool.map(torch.load, [data_dir + '/all_p_img' + '.pt', data_dir + '/all_n_img' + '.pt'])
# all_p_list, all_n_list = pool.map(torch.load, [data_dir + '/all_p_list' + '.pt', data_dir + '/all_n_list' + '.pt'])


hight = 224

a = torch.empty(22199, 3+3+4, hight, hight)
b = torch.empty(22199, 3+3+4, hight, hight)
# c = torch.empty(2000, 512, 9)
# d = torch.empty(2000, 512, 9)

a[:, :, :, :] = all_p_img[:22199, :, :, :]
b[:, :, :, :] = all_n_img[:22199, :, :, :]
# c[:, :, :] = all_p_list[:2000, :, :]
# d[:, :, :] = all_n_list[:2000, :, :]

torch.save(a, data_dir + '/all_p_img' + '.pt')
torch.save(b, data_dir + '/all_n_img' + '.pt')
# torch.save(c, data_dir + '/all_p_list2000' + '.pt')
# torch.save(d, data_dir + '/all_n_list2000' + '.pt')