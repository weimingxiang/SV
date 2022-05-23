import example
import torch

data_dir = "../datasets/NA12878_PacBio_MtSinai/"


mid_sign_list = torch.load(data_dir + "chromosome_sign/" + "chrM" + "_m(i)d_sign.pt")

example.deal_list(mid_sign_list)