"""
Construct and train a two tower recommender.
"""
import torch
from Recommender_2Tower__NNDataset import Dataset2Tower, Dataset2Tower_collate_fn
import Recommender_2Tower__config as cfg
import Recommender_2Tower__data as dq
from Recommender_2Tower__train import train_recommender
from Recommender_2Tower__eval import eval_recommender
from torch.utils.data import DataLoader, random_split
import sys, os, pickle


if __name__ == "__main__":

    # @@@@@@@@@@ Loading & pre-process raw data @@@@@@@@@@ #
    df_inter, df_users, df_items, dict_item_img_text_source = dq.load_data()  # load and define data range

    df_inter, df_users, df_items, df_users_cat_hist, CoN_users_cat_hist = \
        dq.structured_data(df_inter, df_users, df_items)  # process and normalized structured_data

    dict_item_text_embed, item_text_embed_size = dq.generate_item_text_embed(dict_item_img_text_source)  # text_embeddings of items

    # .......................................
    cfg.corpus_size = df_items.shape[0]
    cfg.two_tower_dims['item_text_embed_size'] = item_text_embed_size  # text embedding size

    # @@@@@@@@@@ Data Preparation @@@@@@@@@@ #
    # ...... dataset .......
    CoN_user_id = cfg.dataset_para['CoN_user_id']
    CoN_item_id = cfg.dataset_para['CoN_item_id']

    df_items_keys = df_items[CoN_item_id].tolist()
    cfg.dataset_para['dict_inx2itemID'] = {inx + 1: id for inx, id in enumerate(df_items_keys)}  # 0: unknown id
    cfg.dataset_para['dict_itemID2inx'] = {id: inx for inx, id in cfg.dataset_para['dict_inx2itemID'].items()}
    df_users_keys = df_users[CoN_user_id].tolist()
    cfg.dataset_para['dict_inx2userID'] = {(inx + 1): id for inx, id in enumerate(df_users_keys)}  # 0 for padding
    cfg.dataset_para['dict_userID2inx'] = {id: inx for inx, id in cfg.dataset_para['dict_inx2userID'].items()}

    cfg.two_tower_dims['user_id_hash_size'] = max(list(cfg.dataset_para['dict_inx2userID'].keys())) + 1  # "num_embeddings = Max_index + 1" for nn.Embedding
    cfg.two_tower_dims['item_id_hash_size'] = max(list(cfg.dataset_para['dict_inx2itemID'].keys())) + 1

    all_dataset = Dataset2Tower(
        dataset_para=cfg.dataset_para,
        df_inter=df_inter,
        df_users=df_users,
        df_users_cat_hist=df_users_cat_hist,
        df_items=df_items,
        dict_item_text_embed=dict_item_text_embed,
        dict_item_img_text_source=dict_item_img_text_source
    )

    all_dataset.gen_feature_dim(cfg.two_tower_dims)

    train_ds, valid_ds, test_ds = random_split(all_dataset, [0.7, 0.1, 0.2],
                                                                generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=Dataset2Tower_collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=Dataset2Tower_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=Dataset2Tower_collate_fn)

    del train_ds
    del valid_ds

    # @@@@@@@@ Train model @@@@@@@@@@ #
    model_recommender = train_recommender(train_loader, valid_loader)

    # @@@@@@@@ Evaluate model @@@@@@@@@@ #
    eval_recommender(df_items, test_loader, dict_item_text_embed, dict_item_img_text_source)

    print("Recommender_2Tower_main: Done!")