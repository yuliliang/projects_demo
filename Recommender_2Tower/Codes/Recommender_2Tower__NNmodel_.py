import numpy as np
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import Recommender_2Tower__config as cfg
import math


# @@@@@@@@@@@@@@ Summarize the historical embeddings by transformer BERT @@@@@@@@@@@@@@@@@@@
class PositionalEncoding(nn.Module):
    """
    Positional Encoding in the transformer.
    """
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)   # [context_len, token_dim]
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 1::2] = torch.cos(position * div_term)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class BERTEncoderFromEmbedding(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, num_layers=4, num_heads=12, dropout=0.1, max_len=512):
        """
        The transformer encoder (i.e., BERT) for summarizing image or text of previous purchased item(s).

        Args:
            input_dim: The input text or image embedding dimension.
            hidden_dim: The dimension of hidden layer in the BERT.
            num_layers: Number of layers.
            num_heads: Number of attention heads.
            dropout: Drop out rate.
            max_len: Maximum context length of this encoder.
        """
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))  # Learnable CLS token, [1,1,384]
        self.pos_encoder = PositionalEncoding(input_dim, max_len=max_len+1)  # length +1 to consider CLS.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.transformer_BERT = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self,
                batch_embs: list[torch.Tensor],
                lengths: list
        )-> torch.tensor:

        # padded a batch, with zeros, in the dimension of "max length of the token in this batch".
        batch_size = len(batch_embs)
        emb_dim = batch_embs[0].shape[1]
        max_len = max(lengths) + 1  # +1 for CLS token

        padded_batch = torch.zeros(batch_size, max_len, emb_dim)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        for i, emb in enumerate(batch_embs):
            seq_len = emb.size(0)
            padded_batch[i, 1:seq_len + 1] = emb  # leave [0] for CLS
            attention_mask[i, :seq_len + 1] = 1  # include CLS in attention

        # insert CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # expand to a batch: [batch_size, 1, emb_dim]
        padded_batch[:, 0:1] = cls_tokens  # insert at position 0

        x = self.pos_encoder(padded_batch)    # add positional encoding
        key_padding_mask = ~attention_mask
        x = self.transformer_BERT(x, src_key_padding_mask=key_padding_mask)

        return x[:, 0]


# @@@@@@@@@@@@@ Search top k items with inner product @@@@@@@@@@@@@@@
class SearchTopK_InnerProduct(nn.Module):
    """
    Search top k items by inner product among embeddings.
    """

    def __init__(
            self,
            corpus_size: int,
            embedding_dim: int
    ) -> None:
        super(SearchTopK_InnerProduct, self).__init__()
        self.corpus_size = corpus_size
        self.embedding_dim = embedding_dim
        self.corpus = torch.tensor(np.array([]))  # A placeholder for all_item_embeddings, [corpus_size, embedding_dim]
        self.corpus_id = torch.tensor(np.array([]))  # A placeholder for corpus id, [corpus_size]

    def forward(
            self,
            query_embedding: torch.Tensor,  # [B, user_2tower_embeddings]
            num_items: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Goal: Find the top num_items items among the embedding inner product

        Args:
            query_embedding: Here is the user embedding from the recommender.
            num_items: Number of top matched items.

        Returns:
            item_ids: id of top k items.
            InProd_scores: Inner product scores.
            embeddings: embeddings of top k items.
        """
        InProd_scores, indices = torch.topk(
            torch.matmul(query_embedding, self.corpus.T),
            k=num_items, dim=1
        )
        # Expand indices to create a 3D tensor [B, num_items, 1]
        expanded_indices = indices.unsqueeze(2)

        item_ids = self.corpus_id[expanded_indices]  # corpus_id: [2991] -->  item_ids = [B, num_items, 1]
        item_ids = item_ids.squeeze(2)  # [B, num_items]

        # Use the expanded indices to gather embeddings from self.corpus
        embeddings = self.corpus[expanded_indices]

        # Squeeze to remove the extra dimension
        embeddings = embeddings.squeeze(2)  # [B, num_items, embed_dim]

        return item_ids, InProd_scores, embeddings


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@ 2-tower Recommender @@@@@@@@@@@@@@@@@@@@@@@@@@@@
class TwoTowerBaseRetrieval(nn.Module):
    """
    Neural Network based Recommender with two tower architecture.
    """
    def __init__(
        self,
        InProd_module: SearchTopK_InnerProduct,    # A module that search top k items by Inner Product Search
        two_tower_dims: dict,
    ) -> None:

        super().__init__()
        self.InProd_module = InProd_module
        self.two_tower_dims = two_tower_dims
        self.num_items = two_tower_dims['num_items']
        context_len = two_tower_dims['context_len']
        user_id_embedding_dim = two_tower_dims['user_id_embedding_dim']
        item_id_embedding_dim = two_tower_dims['item_id_embedding_dim']
        user_feature_embed_dim = two_tower_dims['user_feature_embed_dim']
        item_feature_embed_dim = two_tower_dims['item_feature_embed_dim']
        user_id_hash_size = two_tower_dims['user_id_hash_size']
        user_features_size = two_tower_dims['user_features_size']
        user_cat_hist_features_size = two_tower_dims['user_cat_hist_features_size']
        item_id_hash_size = two_tower_dims['item_id_hash_size']
        item_features_size = two_tower_dims['item_features_size']
        item_text_embed_size = two_tower_dims['item_text_embed_size']
        item_img_embed_size = two_tower_dims['item_img_embed_size']

        text_BERT_hidden_dim = two_tower_dims['text_BERT_hidden_dim']
        img_BERT_hidden_dim = two_tower_dims['img_BERT_hidden_dim']
        text_distill_embed_dim = two_tower_dims['text_distill_embed_dim']
        img_distill_embed_dim = two_tower_dims['img_distill_embed_dim']
        user_item_2tower_dim = two_tower_dims['user_item_2tower_dim']

        # ....... Create the machinery for user tower .......
        # 1. Convert the user id index into embeddings.
        if two_tower_dims['with_user_id_embedding']:
            self.user_id_embedding_arch = nn.Embedding(
                user_id_hash_size, user_id_embedding_dim, padding_idx=0
            )

        # 2-a. Using transformer to summarize the images of previously purchased items.
        self.user_img_BERTencoder = BERTEncoderFromEmbedding(
            input_dim=item_img_embed_size, hidden_dim=img_BERT_hidden_dim,
            num_layers=2, num_heads=4, dropout=0.2, max_len=context_len)
        self.user_img_arch = nn.Sequential(
            nn.ReLU(),
            nn.Linear(item_img_embed_size, img_distill_embed_dim)
        )

        # 2-b. Using transformer to summarize the text (i.e., product descriptions) of previously purchased items.
        self.user_text_BERTencoder = BERTEncoderFromEmbedding(
            input_dim=item_text_embed_size, hidden_dim=text_BERT_hidden_dim,
            num_layers=2, num_heads=4, dropout=0.1, max_len=context_len)
        self.user_text_arch = nn.Sequential(
            nn.ReLU(),
            nn.Linear(item_text_embed_size, text_distill_embed_dim)
        )

        # 3. Process the user_features & user_cat_hist.
        self.user_features_arch = nn.Sequential(
            nn.Linear(user_features_size + user_cat_hist_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, user_feature_embed_dim),
        )

        # 4. Integrate all the user_tower_input (i.e., id, user features, purchased items text & image.)
        if two_tower_dims['with_user_id_embedding']:
            self.user_tower_arch = nn.Linear(
                in_features=(user_id_embedding_dim + user_feature_embed_dim +
                             text_distill_embed_dim + img_distill_embed_dim),
                out_features=user_item_2tower_dim
            )
        else:
            self.user_tower_arch = nn.Linear(
                in_features=(user_feature_embed_dim + text_distill_embed_dim + img_distill_embed_dim),
                out_features=user_item_2tower_dim
            )

        # ....... Create the architectures for item tower .......
        # 1. Convert the item id into embeddings.
        if two_tower_dims['with_item_id_embedding']:
            self.item_id_embedding_arch = nn.Embedding(
                item_id_hash_size, item_id_embedding_dim, padding_idx=0
            )

        # 2-a. Further Process the item images (i.e., product images) into a smaller embeddings.
        self.item_img_arch = nn.Sequential(
            nn.Linear(item_img_embed_size, 256),
            nn.ReLU(),
            nn.Linear(256, img_distill_embed_dim)
        )

        # 2-b. Further Process the item text (i.e., product description) into a smaller embeddings.
        self.item_text_arch = nn.Sequential(
            nn.Linear(item_text_embed_size, 256),
            nn.ReLU(),
            nn.Linear(256, text_distill_embed_dim)
        )

        # 3. Process the item features
        self.item_features_arch = nn.Sequential(
            nn.Linear(item_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, item_feature_embed_dim),
        )

        # 4. Input all the item features and generate item embeddings (i.e., id, item_features, text_embed, & img_embed)
        if two_tower_dims['with_item_id_embedding']:
            self.item_tower_arch = nn.Linear(
                in_features=(item_id_embedding_dim + item_feature_embed_dim +
                             text_distill_embed_dim + img_distill_embed_dim),
                out_features=user_item_2tower_dim
            )
        else:
            self.item_tower_arch = nn.Linear(
                in_features=(item_feature_embed_dim + text_distill_embed_dim + img_distill_embed_dim),
                out_features=user_item_2tower_dim
            )

    def compute_user_embedding(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,    # [B, fea_dim]
        user_cat_hist: torch.Tensor,    # [B, fea_dim]
        user_purchased_items_len: int,  # [B]
        user_purchased_items_text: list[torch.Tensor],  # len=B, Each torch tensor shape = [context_len, text_dim]
        user_purchased_items_img: list[torch.Tensor]   # len=B. Each torch tensor shape = [context_len, img_dim]
    ) -> torch.Tensor:
        """
        Compute the user embeddings in the recommender.

        Args:
            user_id: the user id index.
            user_features: the user features. We are assuming these are all dense features.
                In practice, you will probably want to support sparse embedding features as well.
            user_cat_hist: for each user, the history of items they have interacted with.
                This is a tensor of item ids. Here we are assuming that the history is
                a fixed length, but in practice you will probably want to support variable
                length histories. jagged tensors are a good way to do this.
                This is NOT USED in this implementation. It is handled in a follow on derived class.
            user_purchased_items_len: For eah user, the number of purchased items.
            user_purchased_items_text: For eah user, the text embeddings of previous purchased items.
            user_purchased_items_img: For eah user, the image embeddings of previous purchased items.

        Returns:
            user_embedding: The user embeddings from the user tower in the recommender.
        """
        # Process user features
        user_features_embedding = self.user_features_arch(torch.cat([user_features, user_cat_hist], axis=1))

        user_text_CLS = self.user_text_BERTencoder(user_purchased_items_text, user_purchased_items_len)
        user_text_embedding = self.user_text_arch(user_text_CLS)
        user_img_CLS = self.user_img_BERTencoder(user_purchased_items_img, user_purchased_items_len)
        user_img_embedding = self.user_img_arch(user_img_CLS)

        # Compute the user embedding, with or without user_id embeddings.
        if self.two_tower_dims['with_user_id_embedding']:
            user_id_embedding = self.user_id_embedding_arch(user_id)
            user_embedding = self.user_tower_arch(torch.cat(
                [user_id_embedding, user_features_embedding, user_text_embedding, user_img_embedding], dim=1
            ))
        else:
            user_embedding = self.user_tower_arch(torch.cat(
                [user_features_embedding, user_text_embedding, user_img_embedding], dim=1
            ))

        return user_embedding

    def compute_item_embeddings(
        self,
        item_id: torch.Tensor,          # [B]
        item_features: torch.Tensor,    # [B, fea_dim]
        item_text_embed: torch.Tensor,  # [B, item_text_embed]
        item_img_embed: torch.Tensor    # [B, item_img_embed]
    ) -> torch.Tensor:
        """
        Compute the item embeddings in the recommender.

        Args:
            item_id: Item id index.
            item_features: Item features.
            item_text_embed: Item text embeddings.
            item_img_embed: Item image embeddings.

        Returns:
            item_embedding: The item embeddings of the item tower in the recommender.
        """
        item_features_embedding = self.item_features_arch(item_features)    # item features embedding
        item_text_embedding = self.item_text_arch(item_text_embed)          # item image text embedding
        item_img_embedding = self.item_img_arch(item_img_embed)             # item image embedding

        # Compute the item embedding
        if self.two_tower_dims['with_user_id_embedding']:
            item_id_embedding = self.item_id_embedding_arch(item_id)  # item id embedding
            item_embedding = self.item_tower_arch(torch.cat(
                [item_id_embedding, item_features_embedding, item_text_embedding, item_img_embedding], dim=1
            ))
        else:
            item_embedding = self.item_tower_arch(torch.cat(
                [item_features_embedding, item_text_embedding, item_img_embedding], dim=1
            ))

        return item_embedding

    def save_item_2tower_embeddings(
            self,
            all_item_embeddings: torch.Tensor,
            all_item_id: torch.Tensor
    ) -> int:
        """
        Save the item tower embedding for later query.

        Args:
            all_item_embeddings: The embedding from the item tower.
            all_item_id: The item id for each embedding.
        """
        self.InProd_module.corpus = all_item_embeddings
        self.InProd_module.corpus_id = all_item_id
        return 1

    def forward(
        self,
        user_id: torch.Tensor,          # [B]
        user_features: torch.Tensor,    # [B, fea_dim]
        user_cat_hist: torch.Tensor,    # [B, fea_dim]
        user_purchased_items_len: int,  # [B]
        user_purchased_items_text: list[torch.Tensor],  # [B, context_len, text_embed]
        user_purchased_items_img: list[torch.Tensor]    # [B, context_len, image_embed]
    ) -> torch.Tensor:
        """
        forward function for the inference.
        Compute the user embedding and return the top num_items items.

        Args:
            user_id: User id index.
            user_features: User features.
            user_cat_hist: User history that are normalized histogram based on item category counts.
            user_purchased_items_text: For each user, the text embedding of previously purchased items.
            user_purchased_items_img: For each user, the image embedding of previously purchased items.

        Returns:
            top_items_id: The id of top n items.
        """
        # Compute the user embedding
        user_embedding = self.compute_user_embedding(
            user_id, user_features, user_cat_hist,
            user_purchased_items_len, user_purchased_items_text, user_purchased_items_img
        )

        # Query the InProd_module to get the top num_items items and their embeddings
        top_items_id, _, _ = self.InProd_module(
            query_embedding=user_embedding, num_items=self.num_items
        )

        return top_items_id

    def compute_training_loss(
        self,
        user_embedding: torch.Tensor,   # [B, user_embed_dim]
        item_embeddings: torch.Tensor,  # [B, item_embed_dim]
        item_log_freq: torch.Tensor  # [B]
    ) -> torch.Tensor:
        """
        With the embeddings from user tower and item tower, calculate the training loss.
        Args:
            user_embedding: Embeddings from user tower.
            item_embeddings: Embeddings from item tower.
            item_log_freq: Log frequency of items.

        Returns:
            loss: loss
        """
        # Compute the scores for every pair of user and item
        scores = torch.matmul(F.normalize(user_embedding), F.normalize(item_embeddings).t())  # [B, B]

        # Here I implement the in-batch negatives with log-Q correction.
        # logit, or inner product = s^c(xi, yj) = s(xi, yj) − log(pj)
        if cfg.apply_logQ_correction:
            logQ_correct_mat = torch.ones(len(item_log_freq), 1) * (item_log_freq.unsqueeze(0))   # nx1 * 1xn = nxn
            scores -= logQ_correct_mat

        # Compute softmax loss
        target = torch.arange(scores.shape[0]).to(scores.device)  # [B]
        # Yuli: this is in-barch sampling.
        # For example, for the 1st user, the target (or positive) is the 1st item. Then we set any other items
        # in the batch as negatives for this user.
        # for the 2nd user, the target is the 2nd item, etc.

        loss = F.cross_entropy(input=scores, target=target, reduction="none")  # [B]
        loss = torch.mean(loss)

        return loss

    def forward_train(
        self,
        user_id: torch.Tensor,          # [B]
        user_features: torch.Tensor,    # [B, fea_dim]
        user_cat_hist: torch.Tensor,    # [B, fea_dim]
        item_id: torch.Tensor,          # [B]
        item_features: torch.Tensor,    # [B, fea_dim]
        item_text_embed: torch.Tensor,  # [B, text_embd_dim]
        item_img_embed: torch.Tensor,   # [B, img_embd_dim]
        item_log_freq: torch.Tensor,    # [B]
        user_purchased_items_len: int,  # [B]
        user_purchased_items_text: torch.Tensor,  # [B, context_len, text_embed_dim]
        user_purchased_items_img: torch.Tensor    # [B, context_len, img_embed_dim]
    ) -> float:
        """
        With input data, generate the embeddings from user tower and item towers, and then computes the
        training loss.

        Args:
            user_id: User id index.
            user_features: User features
            user_cat_hist: User history in the form of normalized histogram of item category count.
            item_id: Item id index.
            item_features (torch.Tensor): Item features.
            item_text_embed: Item text (product description) embeddings.
            item_img_embed: Item image (product images) embeddings.
            item_log_freq: Item log frequency.
            user_purchased_items_len: Length of context for the transformer, or number of user purchased items.
            user_purchased_items_text: Items text (product description) embeddings of user purchased items.
            user_purchased_items_img: Items image (product images) embeddings of user purchased items

        Returns:
            loss: The computed loss.
        """
        # Compute the user embedding
        user_embedding = self.compute_user_embedding(
            user_id, user_features, user_cat_hist,
            user_purchased_items_len, user_purchased_items_text, user_purchased_items_img
        )
        # Compute item embeddings
        item_embeddings = self.compute_item_embeddings(
            item_id, item_features, item_text_embed, item_img_embed
        )

        loss = self.compute_training_loss(
            user_embedding=user_embedding,
            item_embeddings=item_embeddings,
            item_log_freq=item_log_freq
        )
        return loss
