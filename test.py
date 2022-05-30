import torchvision
import torch


data_dir = "../datasets/NA12878_PacBio_MtSinai/"


resize = torchvision.transforms.Resize([256, 11])


for index in range(22199):
    a, b = torch.load(data_dir + '/positive_data/' + str(index) + '.pt')
    torch.save([a, torch.squeeze(resize(b.unsqueeze(0)))], data_dir + '/positive_data/' + str(index) + '.pt')
#     index += length

# for i in range(len(n_position)):
    a, b = torch.load(data_dir + '/negative_data/' + str(index) + '.pt')
    torch.save([a, torch.squeeze(resize(b.unsqueeze(0)))], data_dir + '/negative_data/' + str(index) + '.pt')
    print(index)
