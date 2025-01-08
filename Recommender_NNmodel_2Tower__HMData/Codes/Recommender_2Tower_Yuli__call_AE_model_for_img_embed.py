"""
structure could take a look at:
https://www.kaggle.com/alincijov/image-similarity-search-in-pytorch
"""
import torch
import os, sys
from torch.utils.data import DataLoader
import Recommender_2Tower_Yuli__config as cfg
sys.path.append(os.path.abspath(cfg.path_img_code))
from yuli_CNN_AE_base_resnet_NNmodel import AutoEncoder, Dataset4Image
from yuli_CNN_AE_base_resnet_data import record_file_names, get_pre_exist_model
from yuli_CNN_AE_base_resnet_evaluation import EvalModel
device = cfg.device


def extract_img_embedding(img_path_file):
    """
    Goal: Extract image embedding from a previously trained AutoEncoder model.
    Args:
        img_path_file:

    Returns:
        embeddings: Image embeddings.
        List_article_id: A list of corresponding article_id (i.e., product id) for each image (product image) embedding.
    """
    # load image model
    PATH_model_early_stop = cfg.path_img_model + cfg.FName_img_model  # cfg.DIR_out + 'model/' + 'model' + str_out
    img_model, optimizer, scheduler_lr, L_norm_coef, epoch_min, PARA = get_pre_exist_model(PATH_model_early_stop)

    # generate image data loader
    dataset = Dataset4Image(img_path_file)
    full_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=30)
    # for loading image, do not set a large batch_size, which causes large memory usage.

    # generate embedding
    em = EvalModel(device=cfg.device)  # set the object to evaluate model
    embeddings, embedding_FName, List_article_id = em.create_embedding(img_model, full_loader)

    # update the dimension of image embedding.
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    cfg.item_img_embedding_size = embeddings.shape[1]

    return embeddings, List_article_id
