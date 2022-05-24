#include <torch/torch.h>
#include <iostream>
#include <string>

string data_dir = "../datasets/NA12878_PacBio_MtSinai/";

int main()
{
    torch::Tensor tensor = torch::load(data_dir + "chromosome_sign/" + "chr1" + "_mids_sign.pt");
    std::cout << tensor << std::endl;
}