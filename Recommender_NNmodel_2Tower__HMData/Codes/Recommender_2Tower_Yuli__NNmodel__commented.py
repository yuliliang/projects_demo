import numpy as np
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import Recommender_2Tower_Yuli__config as cfg
from Recommender_2Tower_Yuli__KNN_InnerProduct import KNN_InnerProduct
import math

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@ 2-tower model @@@@@@@@@@@@@@@@@@@@@@@@@@@@
class TwoTowerBaseRetrieval(nn.Module):
    """ The 2-tower NN model module """
    def __init__(
        self,
        user_id_hash_size: int,     # size of the embedding table for users
        user_id_embedding_dim: int, # internal dimension
        user_features_size: int,    # input feature size for users
        user_history_features_size: int,  # input feature size for users history features.
        item_id_hash_size: int,     # size of the embedding table for items
        item_id_embedding_dim: int, # internal dimension
        item_features_size: int,    # input feature size for items
        item_text_embed_size: int,  # size of text embedding
        item_img_embedding_size: int,   # image embedding size
        num_items: int,             # number of items to return per user/query
        InProd_module: KNN_InnerProduct    # module that apply KNN by Inner Product Search
    ) -> None:

        super().__init__()
        self.num_items = num_items
        # [T] dimensional vector describing how positive each label is.
        self.InProd_module = InProd_module   # make the item embedding store in another class object.

        # the embedding of item embedding. Here is initialization & it will be updated later.
        self.item_corpus = torch.tensor(np.array([]))  # torch.randn(item_id_hash_size, item_id_embedding_dim)  # [C, DI]

        # --------- Components in user tower ---------
        # Embedding layers for user id. Create a module to represent user preference by a table lookup.
        self.user_id_embedding_arch = nn.Embedding(
            user_id_hash_size, user_id_embedding_dim, padding_idx=0
        )

        # Create an arch to process the user_features
        self.user_features_arch = nn.Sequential(
            nn.Linear(user_features_size + user_history_features_size, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, user_id_embedding_dim)
        )

        # Create an arch to process the user_tower_input
        # Input dimension =
        #   user_id_embedding_dim from get_user_embedding + user_id_embedding_dim from user_features_arch
        # Output dimension = item_id_embedding_dim
        self.user_tower_arch = nn.Linear(
            2 * user_id_embedding_dim, item_id_embedding_dim
        )

        # --------- Components in item tower ---------
        # Embedding layers for item id
        self.item_id_embedding_arch = nn.Embedding(
            item_id_hash_size, item_id_embedding_dim, padding_idx=0
        )
        # Create an arch to process the item_features
        self.item_features_arch = nn.Sequential(
            nn.Linear(item_features_size, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, item_id_embedding_dim)
        )

        # Create an arch to process the item image embeddings.
        self.item_img_arch = nn.Sequential(
            nn.Linear(item_img_embedding_size, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, item_id_embedding_dim)
        )

        # Create an arch to process the item text embeddings (product description).
        self.item_text_arch = nn.Sequential(
            nn.Linear(item_text_embed_size, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, item_id_embedding_dim)
        )

        # 3. Create an arch to process the item_tower_input
        self.item_tower_arch = nn.Linear(
            in_features= 4 * item_id_embedding_dim,  # i.e., concat id + features + text_embed + img_embed
            out_features=item_id_embedding_dim
        )

    def get_user_embedding(
        self,
        user_id: torch.Tensor  # [B]
        #  user_features: torch.Tensor,  # [B, IU]
    ) -> torch.Tensor:
        """
        Goal: Extract user embedding from user_id.
        Args:
            user_id

        Returns:
            user_id_embedding
        """
        user_id_embedding = self.user_id_embedding_arch(user_id)
        return user_id_embedding

    def process_user_features(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
        user_history: torch.Tensor  # [B, H]
    ) -> torch.Tensor:
        """
        Goal: Process the user features to compute the input to user tower arch.

        Args:
            user_id: User IDs. Shape: [B]
            user_features: User features. Shape: [B, IU]
            user_history: For each batch an H length history of ids. Shape: [B, H]

        Returns:
            torch.Tensor: Shape: [B, 2 * DU]
        """
        user_id_embedding = self.get_user_embedding(
            user_id=user_id  # , user_features=user_features
        )  # [B, DU]

        # Process user features
        user_features_embedding = self.user_features_arch(
            torch.cat([user_features, user_history], axis=1)   # concatenate in axis=1
        )  # [B, DU]

        # Concatenate the inputs for compute the user embedding later.
        user_tower_input = torch.cat(
            [user_id_embedding, user_features_embedding], dim=1
        )
        return user_tower_input

    def compute_user_embedding(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
        user_history: torch.Tensor  # [B, H]
    ) -> torch.Tensor:
        """
        Goal: Combine multiple user related features and compute the user embedding.

        Args:
            user_id: User id
            user_features: User features.
            user_history: The feature contain the history of items they have interacted with.

        Returns:
            user_embedding: torch.Tensor with shape: [B, DI]
        """
        user_tower_input = self.process_user_features(
            user_id=user_id, user_features=user_features, user_history=user_history
        )
        # Compute the user embedding
        user_embedding = self.user_tower_arch(user_tower_input)  # [B, DI]
        return user_embedding

    def compute_item_embeddings(
        self,
        item_id: torch.Tensor,  # [B]
        item_features: torch.Tensor,  # [B, II]
        item_text_embed: torch.Tensor,  # [B, text embed]
        item_img_embed: torch.Tensor  # [B, channel x H x W]
    ) -> torch.Tensor:
        """
        Goal: Process item_id and item_features to compute item embeddings.

        Args:
            item_id: Item IDs. Shape: [B]
            item_features: Item features. Shape: [B, II]

        Returns:
            item embeddings: Shape: [B, DI]
        """
        item_id_embedding = self.item_id_embedding_arch(item_id)  # Process item id
        item_features_embedding = self.item_features_arch(item_features)  # Process item features
        item_text_embedding = self.item_text_arch(item_text_embed)  # Process item image text
        item_img_embedding = self.item_img_arch(item_img_embed)  # Process item image embedding

        # Concatenate the inputs and pass them through a linear layer to compute the item embedding
        item_tower_input = torch.cat(
            [item_id_embedding, item_features_embedding, item_text_embedding, item_img_embedding], dim=1
        )
        # Compute the item embedding
        item_embedding = self.item_tower_arch(item_tower_input)  # [B, DI]
        return item_embedding

    def save_all_item_embedding(self,
            all_item_embeddings: torch.Tensor,
            all_item_id: torch.Tensor
    ):
        """
        Goal: Save the pre-calculated item embedding for efficient search later.

        Args:
            all_item_embeddings:
            all_item_id:

        Returns:
            None
        """
        self.InProd_module.corpus = all_item_embeddings
        self.InProd_module.corpus_id = all_item_id
        return 1

    def forward(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
        user_history: torch.Tensor  # [B, H]
    ) -> torch.Tensor:
        """
        Goal: Compute the user embedding and search the top num_items items with the InProd_module module.
        This is used in the inference

        Args:
            user_id: User ID. Shape: [B]
            user_features: User features. Shape: [B, IU]
            user_history: User history. Shape: [B, H]

        Returns:
            top_items_inx: Top num_items items. Shape: [B, num_items]
        """
        # Compute the user embedding
        user_embedding = self.compute_user_embedding(
            user_id, user_features, user_history
        )
        # Query the InProd_module to get the top num_items items and their embeddings
        top_items_inx, _, _ = self.InProd_module(
            query_embedding=user_embedding, num_items=self.num_items
        )  # Returns indices [B, num_items], scores, embeddings

        return top_items_inx

    def compute_training_loss(
        self,
        user_embedding: torch.Tensor,  # [B, DI]
        item_embeddings: torch.Tensor,  # [B, DI]
        item_log_freq: torch.Tensor  # [B]
    ) -> torch.Tensor:
        """
        Goal: The sub function that compute training loss.

        Args:
            user_embedding:
            item_embeddings:
            item_log_freq:

        Returns:
            loss: train loss
        """
        scores = torch.matmul(F.normalize(user_embedding), F.normalize(item_embeddings).t())  # [B, B]

        if cfg.apply_logQ_correction:
            logQ_correct_mat = torch.ones(len(item_log_freq), 1) * (item_log_freq.unsqueeze(0))
            # nx1 * 1xn = nxn # logQ (log-Q) correction
            # in the above "scores", the item is 1 column an item. so the log frequency should be 1 frequency for each column
            scores -= logQ_correct_mat     # logQ correction: s -= log(item_frequency)

        target = torch.arange(scores.shape[0]).to(scores.device)  # [B]
        loss = F.cross_entropy(input=scores, target=target, reduction="none")  # [B]
        loss = torch.mean(loss)
        return loss

    def forward_train(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
        user_history: torch.Tensor,  # [B, H]
        item_id: torch.Tensor,  # [B]
        item_features: torch.Tensor,  # [B, II]
        item_text_embed: torch.Tensor,
        item_img_embed: torch.Tensor,  # [B, channel x H x W]
        item_log_freq: torch.Tensor  # [B]
    ) -> float:
        """
        Goal: The major function computes training loss.

        Args:
            user_id: User IDs. Shape: [B].
            user_features: User features. Shape: [B, IU].
            user_history: User history. Shape: [B, H].
            item_id: Item IDs. Shape: [B].
            item_features: Item features. Shape: [B, II].
            item_text_embed: Product description embedding.
            item_img_embed: Product image embedding,  # [B, channel x H x W]
            item_log_freq: log(item_frequency),  # [B]

        Returns:
            loss: The training loss.
        """
        # Compute the user embedding
        user_embedding = self.compute_user_embedding(
            user_id, user_features, user_history
        )  # [B, DI]
        # Compute item embeddings
        item_embeddings = self.compute_item_embeddings(
            item_id, item_features, item_text_embed, item_img_embed
        )  # [B, DI]

        loss = self.compute_training_loss(
            user_embedding=user_embedding,
            item_embeddings=item_embeddings,
            item_log_freq=item_log_freq
        )
        return loss  # ()


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@ Dataset @@@@@@@@@@@@@@@@@@@@@@@@@@@@
class CustomDataset(Dataset):
    """
    Dataset for the 2 Tower model
    """
    def __init__(self, df_inter,
                 CoN_user_id_inx, df_users, l_user_features, df_user_history, l_user_history_features,
                 CoN_item_id_inx, df_items, l_item_features, dict_item_text_embed,
                 L_img_embed, L_img_item_id_inx,
                 l_labels
                 ):

        # make sure the data type are align in date_time.
        self.CoN_t_dat = 't_dat'
        df_inter[self.CoN_t_dat] = pd.to_datetime(df_inter[self.CoN_t_dat])
        df_user_history[self.CoN_t_dat] = pd.to_datetime(df_user_history[self.CoN_t_dat])

        self.df_inter = df_inter
        self.CoN_user_id_inx = CoN_user_id_inx
        self.df_users = df_users
        self.l_user_features = l_user_features
        self.df_user_history = df_user_history
        self.l_user_history_features = l_user_history_features
        self.CoN_item_id_inx = CoN_item_id_inx
        self.df_items = df_items
        self.l_item_features = l_item_features
        self.dict_item_text_embed = dict_item_text_embed
        self.labels = l_labels
        self.L_img_embed = L_img_embed
        self.L_img_item_id_inx = L_img_item_id_inx

        datasize = df_inter.shape[0]
        dict_item_count = df_inter[CoN_item_id_inx].value_counts().to_dict()

        # item frequency: The following 2 approaches are both working.
        # The 2nd one seems produce better accuracy. But 3000 might be a small dataset to have a conclusion.
        # self.dict_item_log_freq = {key: math.log(val/datasize) for key, val in dict_item_count.items()}
        self.dict_item_log_freq = {key: math.log(val) for key, val in dict_item_count.items()}

    def __len__(self):
        return len(self.df_inter)

    def __getitem__(self, idx):
        t_dat = self.df_inter.iloc[idx, :][self.CoN_t_dat]
        user_id_inx = self.df_inter.iloc[idx, :][self.CoN_user_id_inx]
        user_features = self.df_users.loc[self.df_users[self.CoN_user_id_inx] == user_id_inx, self.l_user_features]
        item_id_inx = self.df_inter.iloc[idx, :][self.CoN_item_id_inx]
        item_features = self.df_items.loc[self.df_items[self.CoN_item_id_inx] == item_id_inx, self.l_item_features]

        user_history = self.df_user_history.loc[ \
            (self.df_user_history[self.CoN_user_id_inx] == user_id_inx) & (self.df_user_history[self.CoN_t_dat] == t_dat),\
            self.l_user_history_features]

        if item_id_inx in self.dict_item_text_embed.keys():
            item_text_embed = self.dict_item_text_embed[item_id_inx]   # already a Torch.Tensor
        else:
            item_text_embed = torch.tensor(np.zeros(cfg.item_text_embed_size))

        item_img_embed = self.L_img_embed[self.L_img_item_id_inx.index(item_id_inx)]
        labels = self.df_inter.iloc[idx, :][self.labels].astype('float32')

        if item_id_inx in self.dict_item_log_freq.keys():
            item_log_freq = torch.tensor(self.dict_item_log_freq[item_id_inx])  # already a Torch.Tensor
        else:
            item_log_freq = torch.tensor(0)

        # make sure the type of user_id is int or number, instead of object.
        # If the return shape is (x), then the batch will return data with shape = (B, x)
        # for user_id, later will be combined to a array list, like np.array([1,2,3]) to input to nn.Embedding.
        return torch.tensor(user_id_inx, dtype=torch.int64), \
            torch.tensor(user_features.to_numpy().reshape(-1), dtype=torch.float32), \
            torch.tensor(user_history.to_numpy().reshape(-1), dtype=torch.float32), \
            torch.tensor(item_id_inx, dtype=torch.int64), \
            torch.tensor(item_features.to_numpy().reshape(-1), dtype=torch.float32), \
            item_text_embed.float(), \
            item_img_embed.float(), \
            torch.tensor(labels.to_numpy(), dtype=torch.float32), \
            item_log_freq


class Dataset_item(Dataset):
    """
    The data set function that only for item data. This is for pre-build the item embedding corpus for KNN search.
    """
    def __init__(self,
                 CoN_item_id_inx, df_items, l_item_features, dict_item_text_embed,
                 L_img_embed, L_img_item_id_inx
                 ):

        self.CoN_item_id_inx = CoN_item_id_inx
        self.df_items = df_items
        self.l_item_features = l_item_features
        self.dict_item_text_embed = dict_item_text_embed
        self.L_img_embed = L_img_embed
        self.L_img_item_id_inx = L_img_item_id_inx

    def __len__(self):
        return len(self.df_items)

    def __getitem__(self, idx):
        item_id_inx = self.df_items.iloc[idx, :][self.CoN_item_id_inx]
        item_features = self.df_items.iloc[idx, :][self.l_item_features]

        if item_id_inx in self.dict_item_text_embed.keys():
            item_text_embed = self.dict_item_text_embed[item_id_inx]   # already a Torch.Tensor
        else:
            item_text_embed = torch.tensor(np.zeros(cfg.item_text_embed_size))

        item_img_embed = self.L_img_embed[self.L_img_item_id_inx.index(item_id_inx)]

        return torch.tensor(item_id_inx, dtype=torch.int64), \
            torch.tensor(item_features.to_numpy().reshape(-1), dtype=torch.float32), \
            item_text_embed.float(), \
            item_img_embed.float()
