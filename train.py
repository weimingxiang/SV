import utilities as ut
from pudb import set_trace
import pandas as pd
import random
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
import os
from net import IDENet
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from multiprocessing import Pool, cpu_count
import pysam
import time
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest import Repeater
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback


os.environ["CUDA_VISIBLE_DEVICES"] = "1, 3"


seed_everything(2022)

data_dir = "../datasets/NA12878_PacBio_MtSinai/"

bam_path = data_dir + "sorted_final_merged.bam"

vcf_filename = data_dir + "insert_result_data.csv.vcf"

all_enforcement_refresh = 0
position_enforcement_refresh = 0
img_enforcement_refresh = 0
sign_enforcement_refresh = 0 # attention
cigar_enforcement_refresh = 0

# get chr list
sam_file = pysam.AlignmentFile(bam_path, "rb")
chr_list = sam_file.references
chr_length = sam_file.lengths
sam_file.close()

hight = 224

if os.path.exists(data_dir + '/all_p_img' + '.pt') and not all_enforcement_refresh:
    pool = Pool(2)
    print("loading")
    # all_p_img = torch.load(data_dir + '/all_p_img' + '.pt')
    # all_n_img = torch.load(data_dir + '/all_n_img' + '.pt')
    all_p_img, all_n_img = pool.map(torch.load, [data_dir + '/all_p_img' + '.pt', data_dir + '/all_n_img' + '.pt'])
    # all_positive_img_i_list = torch.load(data_dir + '/all_p_list' + '.pt')
    # all_negative_img_i_list = torch.load(data_dir + '/all_n_list' + '.pt')
    all_p_list, all_n_list = pool.map(torch.load, [data_dir + '/all_p_list' + '.pt', data_dir + '/all_n_list' + '.pt'])

    pool.close()
    print("loaded")
else:
    all_positive_img = torch.empty(0, 3, hight, hight)
    all_negative_img = torch.empty(0, 3, hight, hight)

    all_positive_img_mid = torch.empty(0, 3, hight, hight)
    all_negative_img_mid = torch.empty(0, 3, hight, hight)

    all_positive_cigar_img = torch.empty(0, 7, hight, hight)
    all_negative_cigar_img = torch.empty(0, 7, hight, hight)

    all_p_list = torch.empty(0, 512, 9)
    all_n_list = torch.empty(0, 512, 9)

    # pool = Pool(2)
    for chromosome, chr_len in zip(chr_list, chr_length):
        print("======= deal " + chromosome + " =======")

        print("position start")
        if os.path.exists(data_dir + 'position/' + chromosome + '/positive' + '.pt') and not position_enforcement_refresh:
            print("loading")
            p_position = torch.load(data_dir + 'position/' + chromosome + '/positive' + '.pt')
            n_position = torch.load(data_dir + 'position/' + chromosome + '/negative' + '.pt')
        else:
            p_position = []
            n_position = []
            insert_result_data = pd.read_csv(vcf_filename, sep = "\t", index_col=0)
            insert_chromosome = insert_result_data[insert_result_data["CHROM"] == chromosome]
            row_pos = []
            row_end = []
            for index, row in insert_chromosome.iterrows():
                row_pos.append(row["POS"])
                row_end.append(row["END"])

            set_pos = set()
            for pos in row_pos:
                set_pos.update(range(pos - 50, pos + 50))

            for pos, end in zip(row_pos, row_end):
                gap = int((end - pos) / 4)
                if gap == 0:
                    gap = 1
                # positive
                begin = pos - 1 - gap
                end = end - 1 + gap
                if begin < 0:
                    begin = 0
                if end >= chr_len:
                    end = chr_len - 1

                p_position.append([begin, end])

                #negative
                insert_length = end - begin
                end = begin
                while end - begin < insert_length / 2 + 1:
                    random_begin = random.randint(1, chr_len)
                    while random_begin in set_pos:
                        random_begin = random.randint(1, chr_len)
                    begin = random_begin - 1 - gap
                    end = begin + insert_length
                    if begin < 0:
                        begin = 0
                    if end >= chr_len:
                        end = chr_len - 1


                n_position.append([begin, end])


            save_path = data_dir + 'position/' + chromosome
            ut.mymkdir(save_path)
            torch.save(p_position, save_path + '/positive' + '.pt')
            torch.save(n_position, save_path + '/negative' + '.pt')
        print("position end")

        print("img start")
        if os.path.exists(data_dir + 'image/' + chromosome + '/positive_img' + '.pt') and not img_enforcement_refresh:
            print("loading")
            # pool = Pool()
            # t_positive_img, t_negative_img = pool.map(torch.load, [data_dir + 'image/' + chromosome + '/positive_img' + '.pt', data_dir + 'image/' + chromosome + '/negative_img' + '.pt'])
            # pool.close()
            t_positive_img = torch.load(data_dir + 'image/' + chromosome + '/positive_img' + '.pt')
            t_negative_img = torch.load(data_dir + 'image/' + chromosome + '/negative_img' + '.pt')
            positive_img_mid = torch.load(data_dir + 'image/' + chromosome + '/positive_img_mid' + '.pt')
            negative_img_mid = torch.load(data_dir + 'image/' + chromosome + '/negative_img_mid' + '.pt')
            positive_img_i = torch.load(data_dir + 'image/' + chromosome + '/positive_img_m(i)d' + '.pt')
            negative_img_i = torch.load(data_dir + 'image/' + chromosome + '/negative_img_m(i)d' + '.pt')


        # if os.path.exists(data_dir + 'image_rd/' + chromosome + '/positive_img' + '.pt') and not enforcement_refresh:
        #     print("loading")
        #     _positive_img, _negative_img = pool.map(torch.load, [data_dir + 'image_rd/' + chromosome + '/positive_img' + '.pt', data_dir + 'image_rd/' + chromosome + '/negative_img' + '.pt'])
        #     print("load end")

        else:
            # chromosome_sign
            if os.path.exists(data_dir + "chromosome_sign/" + chromosome + ".pt") and not sign_enforcement_refresh:
                chromosome_sign = torch.load(data_dir + "chromosome_sign/" + chromosome + ".pt")
                mid_sign = torch.load(data_dir + "chromosome_sign/" + chromosome + "_mids_sign.pt")
                mid_sign_img = torch.load(data_dir + "chromosome_img/" + chromosome + "_m(i)d_sign.pt")
            else:
                ut.mymkdir(data_dir + "chromosome_sign/")
                chromosome_sign, mid_sign, mid_sign_list = ut.preprocess(bam_path, chromosome, chr_len, data_dir)
                torch.save(chromosome_sign, data_dir + "chromosome_sign/" + chromosome + ".pt")
                torch.save(mid_sign, data_dir + "chromosome_sign/" + chromosome + "_mids_sign.pt")
                torch.save(mid_sign_list, data_dir + "chromosome_sign/" + chromosome + "_m(i)d_sign.pt")
                mid_sign_img = ut.mid_list2img(mid_sign_list)
                ut.mymkdir(data_dir + "chromosome_img/")
                torch.save(mid_sign_img, data_dir + "chromosome_img/" + chromosome + "_m(i)d_sign.pt")
            #f # cigar
            # if os.path.exists(data_dir + "chromosome_cigar/" + chromosome + ".pt") and not cigar_enforcement_refresh:
            #     chromosome_cigar, chromosome_cigar_len, refer_q_table = torch.load(data_dir + "chromosome_cigar/" + chromosome + ".pt")
            # else:
            #     ut.mymkdir(data_dir + "chromosome_cigar/")
            #     chromosome_cigar, chromosome_cigar_len, refer_q_table = ut.preprocess_cigar(bam_path, chromosome)
            #     torch.save([chromosome_cigar, chromosome_cigar_len, refer_q_table], data_dir + "chromosome_cigar/" + chromosome + ".pt")
            #     # torch.save(chromosome_cigar, data_dir + "chromosome_cigar/" + chromosome + ".pt")

            rd_depth_mean = torch.mean(chromosome_sign[2].float())

            positive_img = [[] for _ in range(len(p_position))]
            negative_img = [[] for _ in range(len(n_position))]
            positive_img_mid = torch.empty(len(p_position), 3, hight, hight)
            negative_img_mid = torch.empty(len(n_position), 3, hight, hight)
            positive_img_i = torch.empty(len(p_position), 512, 9)
            negative_img_i = torch.empty(len(n_position), 512, 9)


            # insert_chromosome = insert_result_data[insert_result_data["CHROM"] == chromosome]
            # for index, row in insert_chromosome.iterrows():
            #     gap = int((row["END"] - row["POS"]) / 4)
            #     if gap == 0:
            #         gap = 1
            #     # positive
            #     begin = row["POS"] - 1 - gap
            #     end = row["END"] - 1 + gap
            #     if begin < 0:
            #         begin = 0
            #     if end >= len(rd_depth):
            #         end = len(rd_depth) - 1
            #     positive_img.append(chromosome_sign[:, begin:end])
            #     #f positive_cigar_img = torch.cat((positive_cigar_img, ut.cigar_img(chromosome_cigar, chromosome_cigar_len, refer_q_table[begin], refer_q_table[end]).unsqueeze(0)), 0)
            #     positive_cigar_img = torch.cat((positive_cigar_img, ut.cigar_img_single(bam_path, chromosome, begin, end).unsqueeze(0)), 0)

            #     #negative
            #     random_begin = random.randint(1,len(rd_depth))
            #     while random_begin == row["POS"]:
            #         random_begin = random.randint(1,len(rd_depth))
            #     begin = random_begin - 1 - gap
            #     end = begin + row["END"] - row["POS"] + 2 * gap
            #     if begin < 0:
            #         begin = 0
            #     if end >= len(rd_depth):
            #         end = len(rd_depth) - 1
            #     negative_img.append(chromosome_sign[:, begin:end])
            #     #f negative_cigar_img = torch.cat((negative_cigar_img, ut.cigar_img(chromosome_cigar, chromosome_cigar_len, refer_q_table[begin], refer_q_table[end]).unsqueeze(0)), 0)
            #     negative_cigar_img = torch.cat((negative_cigar_img, ut.cigar_img_single(bam_path, chromosome, begin, end).unsqueeze(0)), 0)

            resize = torchvision.transforms.Resize([512, 9])

            for i, b_e in enumerate(p_position):
                positive_img[i] = chromosome_sign[:, b_e[0]:b_e[1]] # dim 3
                positive_img_mid[i] = ut.to_img_mid_single(mid_sign[:, b_e[0]:b_e[1]]) # dim 3
                positive_img_i[i] = resize(mid_sign_img[b_e[0]:b_e[1]].unsqueeze(0))
                print("===== finish(positive_img) " + chromosome + " " + str(i))


            for i, b_e in enumerate(n_position):
                negative_img[i] = chromosome_sign[:, b_e[0]:b_e[1]]
                negative_img_mid[i] = ut.to_img_mid_single(mid_sign[:, b_e[0]:b_e[1]]) # dim 3
                negative_img_i[i] = resize(mid_sign_img[b_e[0]:b_e[1]].unsqueeze(0))

                print("===== finish(negative_img) " + chromosome + " " + str(i))


            # _positive_img, _negative_img = pool.starmap(ut.to_input_image, zip([positive_img, negative_img], [rd_depth_mean] * 2))
            t_positive_img = ut.to_input_image(positive_img, rd_depth_mean)
            t_negative_img = ut.to_input_image(negative_img, rd_depth_mean)
            print("save image start")

            save_path = data_dir + 'image/' + chromosome

            ut.mymkdir(save_path)
            # pool.starmap(torch.save, zip([_positive_img, _negative_img, positive_cigar_img, negative_cigar_img], [save_path + '/positive_img' + '.pt', save_path + '/negative_img' + '.pt', save_path + '/positive_cigar_img' + '.pt', save_path + '/negative_cigar_img' + '.pt']))
            torch.save(t_positive_img, save_path + '/positive_img' + '.pt')
            torch.save(t_negative_img, save_path + '/negative_img' + '.pt')
            torch.save(positive_img_mid, save_path + '/positive_img_mid' + '.pt')
            torch.save(negative_img_mid, save_path + '/negative_img_mid' + '.pt')
            torch.save(positive_img_i, save_path + '/positive_img_m(i)d' + '.pt')
            torch.save(negative_img_i, save_path + '/negative_img_m(i)d' + '.pt')
        print("img end")

        # img/positive_cigar_img
        print("cigar start")
        if os.path.exists(data_dir + 'image/' + chromosome + '/positive_cigar_img' + '.pt') and not cigar_enforcement_refresh:
            print("loading")
            positive_cigar_img = torch.load(data_dir + 'image/' + chromosome + '/positive_cigar_img' + '.pt')
            negative_cigar_img = torch.load(data_dir + 'image/' + chromosome + '/negative_cigar_img' + '.pt')
            # 由于未刷新数据增加的代码
            # all_p_img0 = positive_cigar_img[:, 0, :, :] + positive_cigar_img[:, 5, :, :]
            # all_n_img0 = negative_cigar_img[:, 0, :, :] + negative_cigar_img[:, 5, :, :]
            # positive_cigar_img = torch.cat([all_p_img0.unsqueeze(1), positive_cigar_img[:, 1:3, :, :]], dim = 1)
            # negative_cigar_img = torch.cat([all_n_img0.unsqueeze(1), negative_cigar_img[:, 1:3, :, :]], dim = 1)
            # save_path = data_dir + 'image/' + chromosome
            # torch.save(positive_cigar_img, save_path + '/positive_cigar_img' + '.pt')
            # torch.save(negative_cigar_img, save_path + '/negative_cigar_img' + '.pt')
            # end 从头跑程序需注释
        else:
            sam_file = pysam.AlignmentFile(bam_path, "rb")
            positive_cigar_img = torch.empty(len(p_position), 4, hight, hight)
            negative_cigar_img = torch.empty(len(n_position), 4, hight, hight)
            for i, b_e in enumerate(p_position):
                #f positive_cigar_img = torch.cat((positive_cigar_img, ut.cigar_img(chromosome_cigar, chromosome_cigar_len, refer_q_table[begin], refer_q_table[end]).unsqueeze(0)), 0)
                try:
                    positive_cigar_img[i] = ut.cigar_new_img_single_optimal(sam_file, chromosome, b_e[0], b_e[1])
                except Exception as e:
                    print(e)
                    print("Exception cigar_img_single_optimal")
                    fail = 1
                    while fail:
                        try:
                            fail = 0
                            positive_cigar_img[i] = ut.cigar_new_img_single_memory(sam_file, chromosome, b_e[0], b_e[1])
                        except Exception as e:
                            fail = 1
                            print(e)
                            print("Exception cigar_new_img_single_memory")
                            time.sleep(5)
                #     try:
                #         positive_cigar_img[i] = ut.cigar_img_single_optimal_time2sapce(sam_file, chromosome, b_e[0], b_e[1])
                #     except Exception as e:
                #         print(e)
                #         print("Exception cigar_img_single_optimal_time2sapce")
                #         try:
                #             positive_cigar_img[i] = ut.cigar_img_single_optimal_time3sapce(sam_file, chromosome, b_e[0], b_e[1])
                #         except Exception as e:
                #             print(e)
                #             print("Exception cigar_img_single_optimal_time3sapce")
                #             positive_cigar_img[i] = ut.cigar_img_single_optimal_time6sapce(sam_file, chromosome, b_e[0], b_e[1])



                print("===== finish(p_position) " + chromosome + " " + str(i))

            for i, b_e in enumerate(n_position):
                #f negative_cigar_img = torch.cat((negative_cigar_img, ut.cigar_img(chromosome_cigar, chromosome_cigar_len, refer_q_table[begin], refer_q_table[end]).unsqueeze(0)), 0)

                try:
                    negative_cigar_img[i] = ut.cigar_new_img_single_optimal(sam_file, chromosome, b_e[0], b_e[1])
                except Exception as e:
                    print(e)
                    print("Exception cigar_img_single_optimal")
                    fail = 1
                    while fail:
                        try:
                            fail = 0
                            negative_cigar_img[i] = ut.cigar_new_img_single_memory(sam_file, chromosome, b_e[0], b_e[1])
                        except Exception as e:
                            fail = 1
                            print(e)
                            print("Exception cigar_new_img_single_memory")
                            time.sleep(60)

                    # try:
                    #     negative_cigar_img[i] = ut.cigar_img_single_optimal_time2sapce(sam_file, chromosome, b_e[0], b_e[1])
                    # except Exception as e:
                    #     print(e)
                    #     print("Exception cigar_img_single_optimal_time2sapce")
                    #     try:
                    #         negative_cigar_img[i] = ut.cigar_img_single_optimal_time3sapce(sam_file, chromosome, b_e[0], b_e[1])
                    #     except Exception as e:
                    #         print(e)
                    #         print("Exception cigar_img_single_optimal_time3sapce")
                    #         negative_cigar_img[i] = ut.cigar_img_single_optimal_time6sapce(sam_file, chromosome, b_e[0], b_e[1])


                print("===== finish(n_position) " + chromosome + " " + str(i))
            sam_file.close()

            save_path = data_dir + 'image/' + chromosome

            torch.save(positive_cigar_img, save_path + '/positive_cigar_img' + '.pt')
            torch.save(negative_cigar_img, save_path + '/negative_cigar_img' + '.pt')
        print("cigar end")

        all_positive_img = torch.cat((all_positive_img, t_positive_img), 0)
        all_negative_img = torch.cat((all_negative_img, t_negative_img), 0)
        all_positive_cigar_img = torch.cat((all_positive_cigar_img, positive_cigar_img), 0)
        all_negative_cigar_img = torch.cat((all_negative_cigar_img, negative_cigar_img), 0)
        all_positive_img_mid = torch.cat((all_positive_img_mid, positive_img_mid), 0)
        all_negative_img_mid = torch.cat((all_negative_img_mid, negative_img_mid), 0)

        all_p_list = torch.cat((all_p_list, positive_img_i), 0)
        # set_trace()
        all_n_list = torch.cat((all_n_list, positive_img_i), 0)


    all_p_img = torch.cat([all_positive_img, all_positive_img_mid, all_positive_cigar_img], 1) # 3, 3, 3
    all_n_img = torch.cat([all_negative_img, all_negative_img_mid, all_negative_cigar_img], 1)

    torch.save(all_p_img, data_dir + '/all_p_img' + '.pt')
    torch.save(all_n_img, data_dir + '/all_n_img' + '.pt')
    torch.save(all_p_list, data_dir + '/all_p_list' + '.pt')
    torch.save(all_n_list, data_dir + '/all_n_list' + '.pt')


my_label = "10+9channel_predict"

logger = TensorBoardLogger(os.path.join("/home/xwm/DeepSVFilter/code", "channel_predict"), name=my_label)

checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints_predict/" + my_label,
    filename='{epoch:02d}-{validation_mean:.2f}-{train_mean:.2f}',
    monitor="validation_loss",
    verbose=False,
    save_last=None,
    save_top_k=5,
    # save_weights_only=True,
    mode="min",
    auto_insert_metric_name=True,
    every_n_train_steps=None,
    train_time_interval=None,
    every_n_epochs=None,
    save_on_train_epoch_end=None,
    every_n_val_epochs=None
)

def main_train():
    config = {
        "lr": 1e-6,
        "batch_size": 12,
        "beta1": 0.9,
        "conv2d_dim_stride": 1,
        "classfication_dim_stride": 400,
    }

    model = IDENet(all_p_img, all_n_img, all_p_list, all_n_list, config)



    resume = "./checkpoints_predict/" + my_label + "/epoch=196-validation_mean=0.73-train_mean=0.98.ckpt"

    trainer = pl.Trainer(
        max_epochs=200,
        gpus=1,
        check_val_every_n_epoch=1,
        # replace_sampler_ddp=False,
        logger=logger,
        # val_percent_check=0,
        callbacks=[checkpoint_callback],
        # resume_from_checkpoint=resume
    )

    trainer.fit(model)


def train_tune(config, checkpoint_dir=None, num_epochs=200, num_gpus=1):
    # config.update(ori_config)
    model = IDENet(all_p_img, all_n_img, config)
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        logger=logger,
        # progress_bar_refresh_rate=0,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model)


def gan_tune(num_samples=250, num_epochs=200, gpus_per_trial=1):
    config = {
        "lr": tune.loguniform(1e-6, 1e-3),
        "batch_size": tune.choice([32, 64]),
        "beta1": tune.uniform(0, 1),
        "conv2d_dim_stride": tune.lograndint(1, 6),
        "classfication_dim_stride": tune.lograndint(1, 997),
    }

    # bayesopt = HyperOptSearch(config, metric="prc_value", mode="max")
    # re_search_alg = Repeater(bayesopt, repeat=5)
    scheduler = ASHAScheduler(
        max_t=num_epochs,
        metric='validation_mean',
        mode='max')

    reporter = CLIReporter(
        parameter_columns=["lr", "batch_size", 'beta1', 'conv2d_dim_stride', "classfication_dim_stride"],
        metric_columns=["train_mean", "validation_mean"])

    analysis = tune.run(
        tune.with_parameters(
            train_tune,
            num_epochs=num_epochs,
        ),
        local_dir="/home/xwm/DeepSVFilter/code/",
        resources_per_trial={
            "cpu": 16,
            "gpu": gpus_per_trial
        },
        metric="validation_mean",
        mode="max",
        config=config,
        num_samples=num_samples,
        # scheduler=scheduler,
        progress_reporter=reporter,
        # search_alg=re_search_alg,
        name="tune_asha")



main_train()
# ray.init(num_cpus=8, num_gpus=2)
# ray.init()
# gan_tune()
