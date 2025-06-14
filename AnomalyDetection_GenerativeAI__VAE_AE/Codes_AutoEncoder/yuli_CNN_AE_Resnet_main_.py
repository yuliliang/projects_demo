"""
Designed and implemented a convolutional Autoencoder (AE) from scratch.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os, pickle, sys, time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from yuli_CNN_AE_Resnet_train import train_with_early_stop
import yuli_CNN_AE_Resnet_config as cfg
from yuli_CNN_AE_Resnet_NNmodel import AutoEncoder
from yuli_CNN_AE_Resnet_NNdataset import Dataset4Image
from yuli_CNN_AE_Resnet_evaluation import EvalModel
from yuli_CNN_AE_Resnet_data import get_pre_exist_model
device = cfg.device

def train_model(data_random_state, all_dataset, writer):
    """
    Goal: Prepare all the necessary steps to train a NN model.

    Args:
        data_random_state: The random number assigned for this round
        all_dataset: Input image dataset.
        writer: Tensorboard writer.

    Returns:
        None. The trained NN model with early stop will be saved into a *.bin in the middle.
    """
    str_out = cfg.str_out + '_' + str(data_random_state)
    epochs = cfg.epochs

    # ==== save parameters and crate necessary folders for save output =======
    # ........ create the folder that for saved data/model/performance plots ......
    for ddd in ['model/', 'storage/', 'check/']:
        if not os.path.isdir(cfg.DIR_out + ddd):
            os.mkdir(cfg.DIR_out + ddd)

    # ----- parameters for models -------
    batch_size_train = cfg.batch_size_train
    lr_optimizer = cfg.lr_optimizer
    L1_reg_coef = cfg.L1_reg_coef
    L2_reg_coef = cfg.L2_reg_coef

    # ...... save model parameters ........
    PARA = {}
    PARA['batch_size_train'] = batch_size_train
    PARA['lr_optimizer'] = lr_optimizer
    PARA['L1_reg_coef'] = L1_reg_coef
    PARA['L2_reg_coef'] = L2_reg_coef
    PARA['epochs'] = epochs
    PARA['str_out'] = str_out
    PARA['hidden_dims'] = cfg.hidden_dims.copy()
    PARA['latent_dims'] = cfg.latent_dims
    PARA['image_dim'] = [cfg.raw_img_channel, cfg.img_h_in, cfg.img_w_in]

    with open(cfg.DIR_out + 'PARA.pkl', 'wb') as f:
        pickle.dump(PARA, f)

    # ========== get data  ============
    # ------ split into train/validation/ test dataset -----
    tmp = len(all_dataset)
    train_ds, validate_ds, test_ds = torch.utils.data.random_split(
        all_dataset, [0.8, 0.05, 0.15], generator=torch.Generator().manual_seed(data_random_state))

    # ------- normalize or standardize the input, when necessary, then fill nan as zero -------
    L_norm_coef = []
    N_batch_train = int(len(train_ds) / batch_size_train)
    PARA['N_batch_train'] = N_batch_train
    train_ldr = DataLoader(dataset=train_ds, shuffle=True, batch_size=batch_size_train,
                           num_workers=cfg.dl_N_workers, pin_memory=cfg.PinM_TF)
    validate_ldr = DataLoader(dataset=validate_ds, shuffle=False, batch_size=batch_size_train,
                              num_workers=cfg.dl_N_workers, pin_memory=cfg.PinM_TF)
    del train_ds

    # ================= training setting ===========================
    model = AutoEncoder(cfg.latent_dims, cfg.hidden_dims, [cfg.raw_img_channel, cfg.img_h_in, cfg.img_w_in])

    if 'gpu_' in cfg.machine_label:
        if torch.cuda.device_count() > 1:
            print("Let's use ", torch.cuda.device_count(), " GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)
    model = model.to(device)
    print(model)
    optimizer = optim.AdamW(model.parameters(), lr=lr_optimizer, betas=(0.9, 0.999), eps=1e-08,
                            weight_decay=0.01, amsgrad=False)
    scheduler_lr = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=lr_optimizer, max_lr=lr_optimizer*10, step_size_up=N_batch_train*4,
        cycle_momentum=False)

    # =============== Train =================
    print('\n@@@@@@@@@@@  Start Training  @@@@@@@@@@@\n')
    train_start_T = time.time()
    train_with_early_stop(writer, model, epochs, optimizer, scheduler_lr, train_ldr, \
                          validate_ldr, str_out, cfg.DIR_out, L_norm_coef, PARA)  # Loss_main, Loss_1_2,
    print('\n@@@@@@@@@@@  Training End  @@@@@@@@@@@')
    print('Training with early stop: ' + str((time.time() - train_start_T)/60) + ' minutes.    Finish at: ' + str(datetime.now()))
    return 1


def eval_saved_model(data_random_state, all_dataset):
    """
    Goal: Evaluate a saved NN model:

    Args:
        data_random_state: The random number assigned for this round
        all_dataset: Input image dataset.

    Returns:
        None. The evaluation results would be saved into files.
    """
    model, _, _, _, _, PARA = get_pre_exist_model(cfg.PATH_model_early_stop)

    # --------- Set object for model evaluation ---------
    em = EvalModel(device=device)

    # ------- reconstruct the image using VAE -----
    em.Reconstruction(model, all_dataset, [100, 4440, 16323, 300, 2045])

    # ------- denoise -----
    em.Denoise(model, all_dataset, [100, 4440, 16323, 300, 2045])

    # ------- Reconstruct images from random noise in latent_space ------
    em.Sampling_from_latent_space(model)

    # -------- Anomaly Detection using Reconstruction Error -----
    em.anomaly_detection(model, all_dataset)

    # -------- Image interpolation ------
    idx_img1, idx_img2 = 508, 4055
    em.Interpolation(model, all_dataset, idx_img1, idx_img2)

    return 1


def eval_saved_model__external_images(data_random_state, FPath_img1, FPath_img2):
    """
    Goal: Evaluate a saved NN model y external images.

    Args:
        data_random_state: The random number assigned for this round

    Returns:
        None. The evaluation results would be saved into files.
    """
    model, _, _, _, _, PARA = get_pre_exist_model(cfg.PATH_model_early_stop)

    # --------- Set evaluation object ---------
    em = EvalModel(device=device)

    # -------- interpolation (external images) ------
    em.Interpolation__external_images(model, FPath_img1, FPath_img2)
    return 1


if __name__ == '__main__':

    # ----- Load data -----
    dataset = Dataset4Image(cfg.DIR_data_file)

    # ----- train / evaluate model based on different random seed -------
    for data_random_state in cfg.L_random_status:

        if cfg.Q_objective == 'train' or cfg.Q_objective == 'train_and_evaluate':
            writer = SummaryWriter(f'./runs/VAE_CNN/random_state_{data_random_state}')
            print('L_random_status = ' + str(data_random_state))
            train_start_T = time.time()
            train_model(data_random_state, dataset, writer)
            ttt = (time.time() - train_start_T) / 60
            print('@@ Train time = ' + str(ttt) + ' minutes.')
            writer.flush()
            writer.close()

        # ===== evaluate saved model ======
        if cfg.Q_objective == 'evaluate' or cfg.Q_objective == 'train_and_evaluate':
            print('L_random_status = ' + str(data_random_state))
            train_start_T = time.time()
            eval_saved_model(data_random_state, dataset)
            ttt = (time.time() - train_start_T) / 60
            print('@@ Evaluation time = ' + str(ttt) + ' minutes.')
            print('-------------------\n')

        print('============= end of one round =============\n')

    print("\n Yuli: Done! ")