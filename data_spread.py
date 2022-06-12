from multiprocessing import Pool, cpu_count
import utilities as ut
import torch

data_dir = "../datasets/NA12878_PacBio_MtSinai/"

# pool = Pool(1)
print("loading data")
# all_p_img = torch.load(data_dir + '/all_p_img' + '.pt')
# all_n_img = torch.load(data_dir + '/all_n_img' + '.pt')

# all_ins_img, all_del_img, all_n_img, all_ins_list, all_del_list, all_n_list = pool.imap(torch.load, [data_dir + '/all_ins_img' + '.pt', data_dir + '/all_del_img' + '.pt', data_dir + '/all_n_img' + '.pt', data_dir + '/all_ins_list' + '.pt', data_dir + '/all_del_list' + '.pt', data_dir + '/all_n_list' + '.pt'])

# all_ins_img = torch.load(data_dir + '/all_ins_img' + '.pt')
# all_del_img = torch.load(data_dir + '/all_del_img' + '.pt')
all_n_img = torch.load(data_dir + '/all_n_img' + '.pt')
all_ins_list = torch.load(data_dir + '/all_ins_list' + '.pt')
all_del_list = torch.load(data_dir + '/all_del_list' + '.pt')
all_n_list = torch.load(data_dir + '/all_n_list' + '.pt')


# all_positive_img_i_list = torch.load(data_dir + '/all_p_list' + '.pt')
# all_negative_img_i_list = torch.load(data_dir + '/all_n_list' + '.pt')
# pool.close()
# pool.join()
print("loaded")

length = len(all_ins_list) + len(all_del_list) + len(all_n_list)

ut.mymkdir(data_dir + "ins/")
ut.mymkdir(data_dir + "del/")
ut.mymkdir(data_dir + "n/")

for index in range(length):
    print(index)
    if index < len(all_ins_list):
        pass
        # image = all_ins_img[index].clone()
        # list = all_ins_list[index].clone()
        # torch.save([{"image" : image, "list" : list}, 2], data_dir + "ins/" + str(index) + ".pt")
    elif index < len(all_ins_list) + len(all_del_list):
        index -= len(all_ins_list)
        # image = all_del_img[index].clone()
        # list = all_del_list[index].clone()
        # torch.save([{"image" : image, "list" : list}, 1], data_dir + "del/" + str(index) + ".pt")

    else:
        index -= len(all_ins_list) + len(all_del_list)
        image = all_n_img[index].clone()
        list = all_n_list[index].clone()
        torch.save([{"image" : image, "list" : list}, 0], data_dir + "n/" + str(index) + ".pt")
