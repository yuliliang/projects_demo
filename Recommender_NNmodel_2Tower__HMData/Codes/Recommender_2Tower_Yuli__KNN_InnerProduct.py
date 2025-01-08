import numpy as np
from typing import Tuple
import torch
import torch.nn as nn


class KNN_InnerProduct(nn.Module):
    """
    Goal: Search K-nearest-neighbor by inner product among embeddings.
    """

    def __init__(
        self,
        corpus_size:int,
        embedding_dim:int
    ) -> None:
        super(KNN_InnerProduct, self).__init__()
        self.corpus_size = corpus_size
        self.embedding_dim = embedding_dim
        # Create a random corpus of size [corpus_size, embedding_dim]
        self.corpus = torch.tensor(np.array([]))
        # place holder, torch.randn(corpus_size, embedding_dim)  # [C, DI]
        # Later we set self.corpus = all_item_embeddings

        self.corpus_id = torch.tensor(np.array([]))
        # place holder, torch.randn(corpus_size, embedding_dim)  # [C, DI]

    def forward(
        self,
        query_embedding: torch.Tensor,  # [B, DI]
        num_items: int  # (NI)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Goal: Find the top num_items items among the embedding inner product
        Args:
            query_embedding:
            num_items:

        Returns:
            item_ids: id of KNN.
            InProd_scores: Inner product scores.
            embeddings: embeddings of KNN.
        """
        InProd_scores, indices = torch.topk(
            torch.matmul(query_embedding, self.corpus.T),
            k=num_items, dim=1
        )
        # Expand indices to create a 3D tensor [B, num_items, 1]
        expanded_indices = indices.unsqueeze(2)

        item_ids = self.corpus_id[expanded_indices]
        # corpus_id: [2991] -->  item_ids = [B, num_items, 1]
        # ps. self.corpus_id[indices] seems also works:
        #(self.corpus_id[indices] != (self.corpus_id[expanded_indices].squeeze(2))).sum() == 0
        item_ids = item_ids.squeeze(2)  # [B, num_items]

        # Use the expanded indices to gather embeddings from self.corpus
        embeddings = self.corpus[expanded_indices]
        # self.corpus: [2991, embed_dim] --> embeddings = [B, num_items, 1, embed_dim]
        # (self.corpus[indices] != (self.corpus[expanded_indices].squeeze(2))).sum() == 0

        # Squeeze to remove the extra dimension
        embeddings = embeddings.squeeze(2)  # [B, num_items, embed_dim]

        return item_ids, InProd_scores, embeddings   # indices [B, NI], InProd_scores, and embeddings [B, NI, DI]