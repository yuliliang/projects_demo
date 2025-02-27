"""
Objective: Train and save a AutoEncoder NN model, based on the ResNet structure, to extract meaningful image embeddings.
    This embedding would be used as a recommender system to extract information from a product image later.

Codes structure:
@ yuli_CNN_AE_base_resnet_config.py
configuration file.

@ yuli_CNN_AE_base_resnet_main.py:
- Train a model and save in the ../output/model/
- contain a function to generate model performance examples (i.e., image reconstruction & KNN results)

@ yuli_CNN_AE_base_resnet_train.py
Codes of train neural network with early stop.

@ yuli_CNN_AE_base_resnet_evaluation.py
Codes of evaluation class "EvalModel"

@ yuli_CNN_AE_base_resnet_NNmodel.py
Codes of Auto Encoder model that based on ResNet, and the defined Dataset class.

@ yuli_CNN_AE_base_resnet_data.py
- Some auxiliary functions related to data processing.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os, pickle, sys, time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
from yuli_CNN_AE_base_resnet_train import train_with_early_stop, Loss_L1_L2
import yuli_CNN_AE_base_resnet_config as cfg
from yuli_CNN_AE_base_resnet_NNmodel import AutoEncoder, Dataset4Image
from yuli_CNN_AE_base_resnet_evaluation import EvalModel
from yuli_CNN_AE_base_resnet_data import record_file_names, ImageManipulation, get_pre_exist_model
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
    sys_stdout = cfg.sys_stdout
    epochs = cfg.epochs

    # ==== save parameters and crate necessary folders for save output =======
    # ........ create the folder that for saved data/model/performance plots ......
    for ddd in ['model/', 'storage/', 'check/']:
        if not os.path.isdir(DIR_out + ddd):
            os.mkdir(DIR_out + ddd)

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

    with open(DIR_out + 'PARA.pkl', 'wb') as f:
        pickle.dump(PARA, f)

    if sys_stdout:
        sys.stdout = open(DIR_out + 'Console' + str_out + '.txt', 'w')  # start to record the console output

    # ========== get data  ============
    # ------ split into train/validation/ test dataset -----
    tmp = len(all_dataset)
    train_ds, validate_ds, test_ds = torch.utils.data.random_split(
        all_dataset, [int(tmp*0.7), int(tmp*0.1), tmp-int(tmp*0.7)-int(tmp*0.1)],
        generator=torch.Generator().manual_seed(data_random_state))

    # device_train_pos, device_validate_pos, device_test_pos = \
    #     np.split(df_device_pos.sample(frac=1, random_state=data_random_state),
    #              [int(.7 * len(df_device_pos)), int(.8 * len(df_device_pos))])

    # ------- normalize or standardize the input, when necessary, then fill nan as zero -------
    L_norm_coef = []
    N_batch_train = int(len(train_ds) / batch_size_train)  # number of batches in the train data
    PARA['N_batch_train'] = N_batch_train
    train_ldr = DataLoader(dataset=train_ds, shuffle=True, batch_size=batch_size_train,
                           num_workers=cfg.dl_N_workers, pin_memory=cfg.PinM_TF)
    validate_ldr = DataLoader(dataset=validate_ds, shuffle=False, batch_size=batch_size_train, # int(len(validate_ds)/1),  # for image, the batch_size can't be too big.
                              num_workers=cfg.dl_N_workers, pin_memory=cfg.PinM_TF)
    del train_ds

    # ================= training setting ===========================
    Loss_main = nn.MSELoss()   # in here, loss = summary image values difference (between encoder input vs. decoder output. Using MSE
    Loss_1_2 = Loss_L1_L2(L1_reg_coef, L2_reg_coef)
    model = AutoEncoder()

    if 'gpu_' in cfg.machine_label:
        if torch.cuda.device_count() > 1:
            print("Let's use ", torch.cuda.device_count(), " GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)
    model = model.to(device)
    print(model)
    optimizer = optim.AdamW(model.parameters(), lr=lr_optimizer, betas=(0.9, 0.999), eps=1e-08,
                            weight_decay=0.01, amsgrad=False)
    # optimizer = optim.Adam(model.parameters(), lr=lr_optimizer,
    #                       weight_decay=L2_reg_coef)  # weight_decay is for L2 regularization
    scheduler_lr = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=lr_optimizer, max_lr=lr_optimizer*10, step_size_up=N_batch_train*4,
        cycle_momentum=False)  # ps. optimizer didn't support cycle_momentum

    # =============== Train =================
    print('\n@@@@@@@@@@@  Start Training  @@@@@@@@@@@\n')
    train_start_T = time.time()
    train_with_early_stop(writer, model, epochs, optimizer, scheduler_lr, Loss_main, Loss_1_2, train_ldr, \
                          validate_ldr, str_out, DIR_out, L_norm_coef, PARA)
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
    str_out = cfg.str_out + '_' + str(data_random_state)
    PATH_model_early_stop = cfg.DIR_out + 'model/' + 'model' + str_out
    model, _, _, _, _, PARA = get_pre_exist_model(PATH_model_early_stop)

    # --------- Evaluation ---------
    # create embedding for all x
    em = EvalModel(device=device)  # set the object to evaluate model

    # ..... compare raw and reconstructed images ......
    for img_idx in [0, 1, 10, 150, 250, 500]:
        em.display_reconstructed_img(model, all_dataset, img_idx)

    batch_size_train = PARA['batch_size_train']
    full_loader = DataLoader(dataset=all_dataset, shuffle=False, batch_size=batch_size_train,
                           num_workers=cfg.dl_N_workers, pin_memory=cfg.PinM_TF)

    # ..... search the knn for each "train" image ......
    embeddings, embedding_FName, List_article_id = em.create_embedding(model, full_loader)
    k_of_knn = 3   # number of neighboring images
    for img_idx in [0, 1, 10, 150, 250, 500]:
        # With a query image, algorithm search KNN among the embeddings, and then plot the corresponding
        # FName in the embedding_FName
        em.knn_image_search(all_dataset, img_idx, k_of_knn, embeddings, embedding_FName, model)
    return 1


if __name__ == '__main__':
    # ----- get data -----
    DIR_data = cfg.DIR_data
    img_path_file = cfg.img_path_file
    DIR_out = cfg.DIR_out

    record_file_names(DIR_data, img_path_file)   # record file names
    dataset = Dataset4Image(img_path_file)  # , transform=transform)

    # ImageManipulation(dataset)

    # ----- train / evaluate model based on different random seed -------
    for data_random_state in cfg.L_random_status:

        if cfg.Q_objective == 'train' or cfg.Q_objective == 'train_evaluate':
            writer = SummaryWriter(f'./runs/CNN_AE/random_state {data_random_state}')
            print('L_random_status = ' + str(data_random_state))
            train_start_T = time.time()
            train_model(data_random_state, dataset, writer)
            ttt = (time.time() - train_start_T) / 60
            print('@@ Train time = ' + str(ttt) + ' minutes.')
            writer.flush()
            writer.close()

        # ===== evaluate saved model ======
        if cfg.Q_objective == 'evaluate' or cfg.Q_objective == 'train_evaluate':
            print('L_random_status = ' + str(data_random_state))
            train_start_T = time.time()
            eval_saved_model(data_random_state, dataset)
            ttt = (time.time() - train_start_T) / 60
            print('@@ Evaluation time = ' + str(ttt) + ' minutes.')
            print('-------------------\n')

        print('============= end of one round =============\n')

    print("\n Yuli: Done! ")