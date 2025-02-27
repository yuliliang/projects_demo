import torch
import torch.nn as nn
import torch.optim as optim
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

import yuli_CNN_AE_base_resnet_config as cfg
from yuli_CNN_AE_base_resnet_NNmodel import AutoEncoder
device = cfg.device


def ImageManipulation(dataset):
    """
    Goal: Display the images in the dataset.

    Args:
        dataset: The dataset that contain the input images
    """
    idx = 0
    img_id = dataset.annotations.iloc[idx, 0]
    img_raw = Image.open(os.path.join(dataset.root_dir, img_id)).convert("RGB")

    # ..... show the original image ......
    transform_2 = transforms.Compose([transforms.ToTensor()])
    img_2 = transform_2(img_raw)
    plt.figure(figsize=(10, 10))
    plt.imshow(img_2.permute(1, 2, 0))
    plt.show()

    # ...... image in the Dataset4Image .......
    img = dataset.transform(img_raw)
    y_label = torch.tensor((dataset.annotations.iloc[idx, 1]))  # .astype(float))
    plt.figure(figsize=(10, 10))
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
    plt.close()
    return 1


def get_pre_exist_model(PATH_model_early_stop):
    """
    Goal: Load a saved NN model.

    Args:
        PATH_model_early_stop: The path + file name of a saved NN model.

    Returns:
        model: The saved trained model.
        optimizer: The optimizer for teh trained model.
        scheduler_lr: The learning rate scheduler when trained the model.
        L_norm_coef: The coefficients when preprocess the input data when training the model.
        epoch_min: The epoch of early stop.
        PARA: Other parameters when training the model.
    """
    print(PATH_model_early_stop)
    if device == "cpu":
        checkpoint = torch.load(PATH_model_early_stop, map_location='cpu')  # load GPU trained model in cpu-only machine
    else:
        checkpoint = torch.load(PATH_model_early_stop)
    PARA = checkpoint['PARA']
    epoch_min = PARA['epoch_min']
    N_batch_train = PARA['N_batch_train']
    lr_optimizer = PARA['lr_optimizer']
    hidden_dims = PARA['hidden_dims']
    latent_dims = PARA['latent_dims']
    image_dim = PARA['image_dim']

    model = AutoEncoder(latent_dims, hidden_dims, image_dim)
    if device != 'cpu':
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)   # https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
    model = model.to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.AdamW(list(model.parameters()), lr=lr_optimizer, betas=(0.9, 0.999), eps=1e-08,
                             weight_decay=0.01, amsgrad=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler_lr = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=lr_optimizer, max_lr=lr_optimizer * 10, step_size_up=N_batch_train * 4,
        cycle_momentum=False)  # because optimizer didn't support cycle_momentum
    scheduler_lr.load_state_dict(checkpoint['scheduler_lr_state_dict'])
    L_norm_coef = checkpoint['L_norm_coef']
    return model, optimizer, scheduler_lr, L_norm_coef, epoch_min, PARA
