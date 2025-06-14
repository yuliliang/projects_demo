
from Recommender_2Tower__NNmodel import TwoTowerBaseRetrieval, SearchTopK_InnerProduct
import Recommender_2Tower__config as cfg
import torch.optim as optim
import torch


def train_recommender(train_loader, valid_loader):
    """
    Train the recommender.

    Args:
        train_loader: Train data loader.
        valid_loader: Valid data loader.
    """

    # # @@@@@@@@ Train and evaluate model@@@@@@@@@@
    # ========= Initiate a recommender model =======
    InProd_module = SearchTopK_InnerProduct(   # the module for search the top-k item embeddings with a user query.
        corpus_size=cfg.corpus_size, embedding_dim=cfg.two_tower_dims['user_item_2tower_dim']
    )

    model = TwoTowerBaseRetrieval(
        InProd_module=InProd_module,
        two_tower_dims=cfg.two_tower_dims,
    )
    model = model.to(cfg.device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)   # 5e-3

    # ========= Train =========
    valid_loss_min = float('inf')
    patient = 0
    tb_step = -1
    for epoch in range(cfg.n_epochs):
        print(f"------ epoch: {epoch} ------")
        model.train()
        L_train_loss = []
        _dl = train_loader
        for i, (user_id_inx, user_features, user_cat_hist, item_id_inx, item_features, item_text_embed, item_img_embed,
                item_log_freq, user_purchased_items_len, user_purchased_items_text, user_purchased_items_img) \
                in enumerate(_dl):

            loss = model.forward_train(
                user_id=user_id_inx,
                user_features=user_features,
                user_cat_hist=user_cat_hist,
                item_id=item_id_inx,
                item_features=item_features,
                item_text_embed=item_text_embed,
                item_img_embed=item_img_embed,
                item_log_freq=item_log_freq,
                user_purchased_items_len=user_purchased_items_len,
                user_purchased_items_text=user_purchased_items_text,
                user_purchased_items_img=user_purchased_items_img
            )
            L_train_loss.append(loss.item())
            tb_step += 1

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Train loss: {sum(L_train_loss) / len(L_train_loss)}")

        # ....... Calculate validation loss ......
        model.eval()
        L_valid_loss = []
        for j, (user_id_inx, user_features, user_cat_hist, item_id_inx, item_features, item_text_embed, item_img_embed,
                item_log_freq, user_purchased_items_len, user_purchased_items_text, user_purchased_items_img) \
                in enumerate(valid_loader):

            loss = model.forward_train(
                user_id=user_id_inx,
                user_features=user_features,
                user_cat_hist=user_cat_hist,
                item_id=item_id_inx,
                item_features=item_features,
                item_text_embed=item_text_embed,
                item_img_embed=item_img_embed,
                item_log_freq=item_log_freq,
                user_purchased_items_len=user_purchased_items_len,
                user_purchased_items_text=user_purchased_items_text,
                user_purchased_items_img=user_purchased_items_img
            )
            L_valid_loss.append(loss.item())

        model.train()
        avg_valid_loss = sum(L_valid_loss) / len(L_valid_loss)
        print(f"Validation loss: {avg_valid_loss}")

        if valid_loss_min > avg_valid_loss:
            valid_loss_min = avg_valid_loss
            torch.save({'model_state_dict': model.state_dict(),
                        'two_tower_dims': cfg.two_tower_dims,
                        'dataset_para': cfg.dataset_para
                        },
                        cfg.TwoTower_model_path)
            patient = 0
        else:
            patient += 1

        print(f"    (patient: {patient})")
        if patient >= cfg.early_stop_patient:
            break   # early stop

    return 1


