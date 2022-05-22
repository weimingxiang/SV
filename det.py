import torch
import torchvision
from pudb import set_trace
set_trace()
model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)

data_dir = "../datasets/NA12878_PacBio_MtSinai/"

t_positive_img = torch.load(data_dir + 'image/' + "chr1" + '/positive_img' + '.pt') # 3

re = model(t_positive_img[:10, :, :112, :112])

re[0]