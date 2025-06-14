import torch, math
from torch.utils.data import Dataset
import pandas as pd
import Recommender_2Tower__config as cfg
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
from sentence_transformers import SentenceTransformer

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@ Dataset for 2 tower recommender @@@@@@@@@@@@@@@@@@@@@@@@@@@@
class Dataset2Tower(Dataset):
    """
    Dataset for the recommender with 2 tower architecture.
    """
    def __init__(self,
                 dataset_para: dict,
                 df_inter: pd.DataFrame,
                 df_users: pd.DataFrame,
                 df_users_cat_hist: pd.DataFrame,
                 df_items: pd.DataFrame,
                 dict_item_text_embed: dict,
                 dict_item_img_text_source: dict
                 ) -> None:

        self.CoN_user_id = CoN_user_id = dataset_para['CoN_user_id']
        self.CoN_item_id = CoN_item_id = dataset_para['CoN_item_id']
        self.CoN_t_dat = CoN_t_dat = dataset_para['CoN_t_dat']
        self.max_seq_len = dataset_para['max_seq_len']
        self.dict_userID2inx = dataset_para['dict_userID2inx']
        self.dict_itemID2inx = dataset_para['dict_itemID2inx']

        df_users_ids = df_users[CoN_user_id].to_list()
        df_users_to_numpy = df_users.loc[:, df_users.columns != CoN_user_id].astype(float).to_numpy()
        self.dict_users = {df_users_ids[inx]: val for inx, val in enumerate(df_users_to_numpy)}

        df_items_ids = df_items[CoN_item_id].to_list()
        df_items_to_numpy = df_items.loc[:, df_items.columns != CoN_item_id].astype(float).to_numpy()
        self.dict_items = {df_items_ids[inx]: val for inx, val in enumerate(df_items_to_numpy)}

        df_inter[CoN_t_dat] = df_inter[CoN_t_dat].astype(str)
        self.df_inter = df_inter[[CoN_user_id, CoN_item_id, CoN_t_dat]]
        self.df_inter_all = self.df_inter.copy()  # retrieve user history information.
        self.df_inter = self.df_inter.iloc[:5000, :]   # test train set

        df_users_cat_hist[CoN_t_dat] = df_users_cat_hist[CoN_t_dat].astype(str)
        df_users_cat_hist__user_id = df_users_cat_hist[CoN_user_id].to_list()
        df_users_cat_hist__t_dat = df_users_cat_hist[CoN_t_dat].to_list()
        df_users_cat_hist_to_numpy = df_users_cat_hist.drop([CoN_user_id, CoN_t_dat], axis=1).astype(float).to_numpy()
        self.dict_user_cat_hist = {(df_users_cat_hist__user_id[inx], df_users_cat_hist__t_dat[inx]): val
                                   for inx, val in enumerate(df_users_cat_hist_to_numpy)}

        # ...... import text model to convert into embeddings .....
        self.dict_item_text_embed = dict_item_text_embed
        self.model_SentTrans = SentenceTransformer('all-MiniLM-L6-v2')
        self.item_text_embed_size = 384

        # ...... import image model to convert into embeddings .....
        self.dict_item_img_text_source = dict_item_img_text_source
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()
        self.model_resnet = model
        self.item_img_embed_size = 2048

        pad_left, pad_top, pad_right, pad_bottom = 43, 0, 43, 1
        self.transform = transforms.Compose([
            transforms.Resize(170),
            transforms.Pad((pad_left, pad_top, pad_right, pad_bottom), fill=0),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # ------ compute sampling frequency for log-Q correction --------
        dict_item_count = self.df_inter_all[CoN_item_id].value_counts().to_dict()
        self.dict_item_log_freq = {key: math.log(val) for key, val in dict_item_count.items()}
        # ------------------------

    def __len__(self):
        return self.df_inter.shape[0]

    def gen_feature_dim(self, two_tower_dims):
        two_tower_dims['user_features_size'] = len(next(iter(self.dict_users.values())))
        two_tower_dims['user_cat_hist_features_size'] = len(next(iter(self.dict_user_cat_hist.values())))
        two_tower_dims['item_features_size'] = len(next(iter(self.dict_items.values())))
        two_tower_dims['item_text_embed_size'] = self.item_text_embed_size
        two_tower_dims['item_img_embed_size'] = self.item_img_embed_size

    def __getitem__(self, idx):
        user_id, item_id, t_dat = self.df_inter.iloc[idx, :].tolist()
        user_id_inx = self.dict_userID2inx[user_id]
        item_id_inx = self.dict_itemID2inx[item_id]
        user_features = self.dict_users[user_id]
        item_features = self.dict_items[item_id]
        user_cat_hist = self.dict_user_cat_hist[(user_id, t_dat)]

        # ....... get list of text and image embeddings for the past purchased items ......
        CoN_user_id, CoN_item_id, CoN_t_dat = self.CoN_user_id, self.CoN_item_id, self.CoN_t_dat

        df_item_time_seq = self.df_inter_all.loc[
            (self.df_inter_all[CoN_user_id] == user_id) & (self.df_inter_all[CoN_t_dat] < t_dat),
            [CoN_item_id, CoN_t_dat]].sort_values(CoN_t_dat, ascending=False)   # the newest item_id rank at the top.

        # get items for transform (most recent "max_seq_len" ones, and the most recent one is the first) with date
        # increase with sequence.
        past_item_id_seq = df_item_time_seq[CoN_item_id].tolist()[:self.max_seq_len]
        past_t_dat_seq = [(pd.to_datetime(t_dat) - pd.to_datetime(i)).days
                                  for i in df_item_time_seq[CoN_t_dat].tolist()][:self.max_seq_len]
        user_purchased_items_days_ago = past_t_dat_seq

        user_items_text_list = []
        if past_item_id_seq:
            for id_ in past_item_id_seq:
                if id_ in self.dict_item_text_embed:
                    user_items_text_list.append(self.dict_item_text_embed[id_])
                elif id_ in self.dict_item_img_text_source:
                    user_items_text_list.append(
                        torch.tensor(self.model_SentTrans.encode(self.dict_item_img_text_source[id_]['detail_desc'])))
                else:  # no text
                    user_items_text_list.append(torch.zeros([self.item_text_embed_size]))
        else:  # no past items
            user_items_text_list.append(torch.zeros([self.item_text_embed_size]))

        user_items_img_list = []
        if past_item_id_seq:
            for id_ in past_item_id_seq:
                if id_ in self.dict_item_img_text_source:
                    img = Image.open(self.dict_item_img_text_source[id_]['img_FName']).convert("RGB")
                    user_items_img_list.append(self.model_resnet(self.transform(img).unsqueeze(0)).squeeze())
                else:  # no image source
                    user_items_img_list.append(torch.zeros([self.item_img_embed_size]))
        else:  # no past items
            user_items_img_list.append(torch.zeros([self.item_img_embed_size]))

        user_purchased_items_text = torch.stack(user_items_text_list, dim=0) \
            if user_items_text_list else torch.zeros([self.item_text_embed_size])
        user_purchased_items_img = torch.stack(user_items_img_list, dim=0) \
            if user_items_img_list else torch.zeros([self.item_img_embed_size])
        # .........................

        if item_id in self.dict_item_text_embed:
            item_text_embed = self.dict_item_text_embed[item_id]
        elif item_id in self.dict_item_img_text_source:
            item_text_embed = \
                torch.tensor(self.model_SentTrans.encode(self.dict_item_img_text_source[item_id]['detail_desc']))
        else:
            item_text_embed = torch.zeros([self.item_text_embed_size])

        if item_id in self.dict_item_img_text_source:
            img = Image.open(self.dict_item_img_text_source[item_id]['img_FName']).convert("RGB")
            item_img_embed = self.model_resnet(self.transform(img).unsqueeze(0)).squeeze()  # [3, 224, 224] --> [2048]
        else:
            item_img_embed = torch.zeros([self.item_img_embed_size])

        item_log_freq = torch.tensor(self.dict_item_log_freq[item_id]) \
            if item_id in self.dict_item_log_freq else torch.tensor(0)

        return torch.tensor(user_id_inx, dtype=torch.long).to(cfg.device), \
            torch.tensor(user_features, dtype=torch.float32).to(cfg.device), \
            torch.tensor(user_cat_hist, dtype=torch.float32).to(cfg.device), \
            torch.tensor(item_id_inx, dtype=torch.long).to(cfg.device), \
            torch.tensor(item_features, dtype=torch.float32).to(cfg.device), \
            item_text_embed.float().to(cfg.device), \
            item_img_embed.float().to(cfg.device), \
            item_log_freq.to(cfg.device), \
            user_purchased_items_text.float().to(cfg.device), \
            user_purchased_items_img.float().to(cfg.device)


def Dataset2Tower_collate_fn(batch):
    """
    The collate_fn for the Dataset2Tower. The major goal of this function is deal with variant number of
    purchased items of each user.
    """
    user_id_inx, user_features, user_cat_hist, item_id_inx, item_features, item_text_embed, \
        item_image_embd, item_log_freq, \
        user_purchased_items_text, user_purchased_items_img = zip(*batch)    # batch: list of tuples

    user_id_inx = torch.stack(user_id_inx, dim=0)
    user_features = torch.stack(user_features, dim=0)
    user_cat_hist = torch.stack(user_cat_hist, dim=0)
    item_id_inx = torch.stack(item_id_inx, dim=0)
    item_features = torch.stack(item_features, dim=0)
    item_text_embed = torch.stack(item_text_embed, dim=0)
    item_image_embd = torch.stack(item_image_embd, dim=0)
    item_log_freq = torch.stack(item_log_freq, dim=0)

    # output list(tensor). Due to the variant length in the BERT token, it can't be stacked into 1 tensor.
    user_purchased_items_text = list(user_purchased_items_text)
    user_purchased_items_img = list(user_purchased_items_img)

    user_purchased_items_len = [len(seq) for seq in user_purchased_items_text]   # list(int)
    return user_id_inx, user_features, user_cat_hist, item_id_inx, item_features, item_text_embed, \
        item_image_embd, item_log_freq, \
        user_purchased_items_len, user_purchased_items_text, user_purchased_items_img


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@ Dataset for pre-compute items 2-tower embedding @@@@@@@@@@@@@@@@@@@@@@@@@@@@
class Dataset__compute_item2TowerEmbed(Dataset):
    """
    Dataset for pre-compute the item embeddings in the recommender.
    """
    def __init__(self,
                 dataset_para: dict,
                 df_items: pd.DataFrame,
                 dict_item_text_embed: dict,
                 dict_item_img_text_source: dict
                 ) -> None:

        self.CoN_item_id = CoN_item_id = dataset_para['CoN_item_id']
        self.max_seq_len = dataset_para['max_seq_len']
        self.dict_itemID2inx = dataset_para['dict_itemID2inx']

        self.df_items_ids = df_items[CoN_item_id].to_list()
        df_items_to_numpy = df_items.loc[:, df_items.columns != CoN_item_id].astype(float).to_numpy()
        self.dict_items = {self.df_items_ids[inx]: val for inx, val in enumerate(df_items_to_numpy)}

        # ...... import text model to convert into embeddings .....
        self.dict_item_text_embed = dict_item_text_embed
        self.model_SentTrans = SentenceTransformer('all-MiniLM-L6-v2')
        self.item_text_embed_size = 384

        # ...... import image model to convert into embeddings .....
        self.dict_item_img_text_source = dict_item_img_text_source
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()
        self.model_resnet = model
        self.item_img_embed_size = 2048

        pad_left, pad_top, pad_right, pad_bottom = 43, 0, 43, 1
        self.transform = transforms.Compose([
            transforms.Resize(170),
            transforms.Pad((pad_left, pad_top, pad_right, pad_bottom), fill=0),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df_items_ids)

    def __getitem__(self, idx):
        item_id = self.df_items_ids[idx]
        item_id_inx = self.dict_itemID2inx[item_id]
        item_features = self.dict_items[item_id]

        if item_id in self.dict_item_text_embed:
            item_text_embed = self.dict_item_text_embed[item_id]
        elif item_id in self.dict_item_img_text_source:
            item_text_embed = \
                torch.tensor(self.model_SentTrans.encode(self.dict_item_img_text_source[item_id]['detail_desc']))
        else:
            item_text_embed = torch.zeros([self.item_text_embed_size])

        if item_id in self.dict_item_img_text_source:
            img = Image.open(self.dict_item_img_text_source[item_id]['img_FName']).convert("RGB")
            item_img_embed = self.model_resnet(self.transform(img).unsqueeze(0)).squeeze()
        else:
            item_img_embed = torch.zeros([self.item_img_embed_size])

        return torch.tensor(item_id, dtype=torch.long).to(cfg.device), \
            torch.tensor(item_id_inx, dtype=torch.long).to(cfg.device), \
            torch.tensor(item_features, dtype=torch.float32).to(cfg.device), \
            item_text_embed.float().to(cfg.device), \
            item_img_embed.float().to(cfg.device), \

