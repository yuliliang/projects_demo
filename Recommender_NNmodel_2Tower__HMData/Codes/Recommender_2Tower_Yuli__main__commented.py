"""
Objective: Train and test a recommender system based on a 2-tower structured Neural Network model.
"""
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import Recommender_2Tower_Yuli__NNmodel__neat as NNmodel
import Recommender_2Tower_Yuli__config as cfg
import Recommender_2Tower_Yuli__data_query as dq
from torch.utils.data import DataLoader, Dataset, random_split
import tqdm, sys, os, pickle
from torch.utils.tensorboard import SummaryWriter
from Recommender_2Tower_Yuli__KNN_InnerProduct import KNN_InnerProduct

sys.path.append(os.path.abspath(cfg.path_img_code))
from yuli_CNN_AE_base_resnet_data import record_file_names

writer = SummaryWriter('./runs/2Tower_HM')   # Writer output to ./runs/ directory by default.

# @@@@@@@@@@ Loading Raw Data & pre-process @@@@@@@@@@ #
# import the available image file in DIR_img_data into file "img_path_file
record_file_names(cfg.DIR_img_data, cfg.img_path_file)

# ..... when process data from scratch. Should take around 2 hours .......
df_items_, df_users_, df_user_history, user_hist_CoN, df_inter_, dict_items_inx2id, dict_users_inx2id\
    = dq.load_and_process_data()  # currently only use top popular product/item with images.
df_items_, df_users_, df_user_history, user_hist_CoN, df_inter_, dict_items_inx2id, dict_users_inx2id, dict_item_text_embed\
    = dq.further_process_data(
    df_items_, df_users_, df_user_history, user_hist_CoN, df_inter_, dict_items_inx2id, dict_users_inx2id)

# ...... load saved data .......
df_items, df_users, df_user_history, user_hist_CoN, df_inter, dict_items_inx2id, dict_users_inx2id \
                    , dict_item_text_embed = dq.further_process_data()   # from saved data.

# tmp = pickle.load(
#     open(cfg.in_file_dir + "SmallerData_3000Items_10000Users__after_further_process_data.bin", "rb"))
# df_items = tmp[0].copy()
# df_users = tmp[1].copy()
# df_user_history = tmp[2].copy()
# user_hist_CoN = tmp[3].copy()
# df_inter = tmp[4].copy()
# dict_items_inx2id = tmp[5].copy()
# dict_users_inx2id = tmp[6].copy()
# dict_item_text_embed = tmp[7].copy()
#
# del tmp

# user_hist_CoN += []
user_hist_CoN = df_user_history.columns.tolist()
for CoN in ['customer_id_inx', 't_dat', 'department_name_sum', 'section_name_sum']:
    user_hist_CoN.remove(CoN)

for CoN in user_hist_CoN:
    df_user_history[CoN] = df_user_history[CoN].astype(float)


# .......... Retrieve image embedding (If prepocessed, the saved file size is 21.84GB) ......
L_img_embed, L_img_item_id_inx = dq.further_process_data_img(dict_items_inx2id)
# df_inter = df_inter.iloc[:3000, :]

# .......
CoN_user_id = cfg.CoN_user_id
CoN_item_id = cfg.CoN_item_id
CoN_user_id_inx = cfg.CoN_user_id_inx
CoN_item_id_inx = cfg.CoN_item_id_inx


# @@@@@@@@@@ Generate negative data & Split Data @@@@@@@@@@ #
ngSize_perUser = cfg.ngSize_perUser   # 80   # for each user, sample ngSize_perUser examples.
df_neg = pd.DataFrame(columns=df_inter.columns)
for user_id in df_inter[CoN_user_id_inx].unique().tolist():
    dft = df_inter.loc[df_inter[CoN_user_id_inx] != user_id].sample(n=ngSize_perUser, random_state=user_id)
    dft[CoN_user_id_inx] = user_id
    df_neg = pd.concat((df_neg,dft), axis=0)
df_neg['price'] = 0

# balance (currently we do this for all. But in reality, negative instance might only be necessary for training).
df_neg = df_neg.sample(frac=1, random_state=42)
df_inter_addNeg = pd.concat([df_inter, df_neg], axis=0)
df_inter_addNeg = df_inter_addNeg.sample(frac=1.0, random_state=42)

# ......... Construct dataset first .........
l_user_features = (df_users.columns).tolist(); l_user_features.remove(CoN_user_id_inx)
l_item_features = (df_items.columns).tolist(); l_item_features.remove(CoN_item_id_inx)
l_labels = [cfg.CoN_target]   # allow multiple labels for later
l_user_history_features = user_hist_CoN.copy()

# ...... construct dataset .......
all_dataset = NNmodel.CustomDataset(
    df_inter,   # df_inter_addNeg,
    CoN_user_id_inx, df_users, l_user_features, df_user_history, l_user_history_features,
    CoN_item_id_inx, df_items, l_item_features, dict_item_text_embed,
    L_img_embed, L_img_item_id_inx,
    l_labels
)

# ......... Split Dataset (with torch.utils.data.random_split) .........
len_dataset = len(all_dataset)
size_train = int(len_dataset * 0.7)
size_valid = int(len_dataset * 0.1)
size_test = len_dataset - size_train - size_valid

train_ds, valid_ds, test_ds = torch.utils.data.random_split(
    all_dataset, [size_train, size_valid, size_test], generator=torch.Generator().manual_seed(42))
# df_train, df_valid, df_test = np.split(df_inter_addNeg, [size_train, size_train+size_valid])

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

# # @@@@@@@@ Train and evaluate model@@@@@@@@@@
# ========= claim the models =======
cfg.corpus_size = df_items.shape[0]
InProd_module = KNN_InnerProduct(   # the module for search the top-k scored item embeddings with a user query.
    corpus_size=cfg.corpus_size, embedding_dim=cfg.item_id_embedding_dim
)

# cfg.num_items,
cfg.user_id_hash_size = df_users['customer_id_inx'].max() + 1  # "num_embeddings = Max_index + 1" for nn.Embedding
# cfg.user_id_embedding_dim,
cfg.user_features_size = len(l_user_features)
cfg.user_history_features_size = len(user_hist_CoN)  # don't count 'customer_id_inx' & 't_dat'
cfg.item_id_hash_size = df_items['article_id_inx'].max() + 1
# cfg.item_id_embedding_dim,
cfg.item_features_size = len(l_item_features)
cfg.item_text_embed_size = len(dict_item_text_embed[list(dict_item_text_embed.keys())[0]])  # pick the 1st key value pair.
cfg.item_img_embedding_size = L_img_embed.shape[1]

model = NNmodel.TwoTowerBaseRetrieval(
    user_id_hash_size=cfg.user_id_hash_size,
    user_id_embedding_dim=cfg.user_id_embedding_dim,
    user_features_size=cfg.user_features_size,
    user_history_features_size=cfg.user_history_features_size,

    item_id_hash_size=cfg.item_id_hash_size,
    item_id_embedding_dim=cfg.item_id_embedding_dim,
    item_features_size=cfg.item_features_size,
    item_text_embed_size=cfg.item_text_embed_size,
    item_img_embedding_size=cfg.item_img_embedding_size,

    num_items=cfg.num_items,
    InProd_module=InProd_module
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
    for i, (user_id, user_features, user_history, item_id, item_features, item_text_embed, item_image_embd, labels, item_log_freq) \
            in enumerate(train_loader):    #
        user_id, user_features, user_history, item_id, item_features, item_text_embed, item_image_embd, labels, item_log_freq = \
            user_id.to(cfg.device), user_features.to(cfg.device), user_history.to(cfg.device), \
            item_id.to(cfg.device),item_features.to(cfg.device), item_text_embed.to(cfg.device), item_image_embd.to(cfg.device),\
            labels.to(cfg.device), item_log_freq.to(cfg.device)

        loss = model.forward_train(
                user_id, # : torch.Tensor,  # [B]
                user_features, # : torch.Tensor,  # [B, IU]
                user_history, # : torch.Tensor,  # [B, H]
                item_id, # : torch.Tensor,  # [B]
                item_features, # : torch.Tensor,  # [B, II]
                item_text_embed,
                item_image_embd,   # : torch.Tensor,  # [B, channel x H x w]
                item_log_freq  # torch.Tensor   # [B]
        )
        L_train_loss.append(loss.item())
        tb_step += 1
        writer.add_scalar("Loss/train", loss.item(), tb_step)
        writer.add_scalar("learning rate", optimizer.param_groups[0]['lr'], tb_step)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Train loss: {sum(L_train_loss) / len(L_train_loss)}")

    # validation loss
    model.eval()
    L_valid_loss = []
    for j, (user_id, user_features, user_history, item_id, item_features, item_text_embed, item_image_embd, labels, item_log_freq) \
            in enumerate(valid_loader):
        user_id, user_features, user_history, item_id, item_features, item_text_embed, item_image_embd, labels, item_log_freq = \
            user_id.to(cfg.device), user_features.to(cfg.device), user_history.to(cfg.device), \
            item_id.to(cfg.device),item_features.to(cfg.device), item_text_embed.to(cfg.device), item_image_embd.to(cfg.device),\
            labels.to(cfg.device), item_log_freq.to(cfg.device)

        loss = model.forward_train(
            user_id,  # : torch.Tensor,  # [B]
            user_features,  # : torch.Tensor,  # [B, IU]
            user_history,  # : torch.Tensor,  # [B, H]
            item_id,  # : torch.Tensor,  # [B]
            item_features,  # : torch.Tensor,  # [B, II]
            item_text_embed,
            item_image_embd,  # : torch.Tensor,  # [B, channel x H x w]
            item_log_freq,   # torch.Tensor  # [B]
        )
        L_valid_loss.append(loss.item())

    model.train()
    avg_valid_loss = sum(L_valid_loss) / len(L_valid_loss)
    writer.add_scalar("Loss/valid", avg_valid_loss, tb_step)
    print(f"Validation loss: {avg_valid_loss}")

    if valid_loss_min > avg_valid_loss:
        valid_loss_min = avg_valid_loss
        torch.save({'model_state_dict': model.state_dict()}, cfg.TwoTower_model_path)
        patient = 0
    else:
        patient += 1

    print(f"    (patient: {patient})")
    if patient >= cfg.early_stop_patient:
        break   # early stop

del model

# ========= test phase =========
# ps. Here we use exact KNN search. In reality we could use ANN.
# def evaluate_model_performance(model):
if True:

    # ...... load the best model from the early stop ......
    model = NNmodel.TwoTowerBaseRetrieval(
        num_items=cfg.num_items,
        user_id_hash_size=cfg.user_id_hash_size,
        user_id_embedding_dim=cfg.user_id_embedding_dim,
        user_features_size=cfg.user_features_size,
        user_history_features_size=cfg.user_history_features_size,

        item_id_hash_size=cfg.item_id_hash_size,
        item_id_embedding_dim=cfg.item_id_embedding_dim,
        item_features_size=cfg.item_features_size,
        item_text_embed_size=cfg.item_text_embed_size,
        item_img_embedding_size=cfg.item_img_embedding_size,

        InProd_module=InProd_module,
    )
    model = model.to(cfg.device)
    model.load_state_dict(torch.load(cfg.TwoTower_model_path, weights_only=True)['model_state_dict'])
    model.eval()

    # ........ pre-calculate the embeddings of all items .........
    all_item_embedding = torch.tensor(np.array([])).float()    # make it be float32, not the default type ".double()".
    all_item_id = torch.tensor(np.array([])).float()

    dataset_item = NNmodel.Dataset_item(
        CoN_item_id_inx, df_items, l_item_features, dict_item_text_embed,
        L_img_embed, L_img_item_id_inx
    )
    loader_item = DataLoader(dataset_item, batch_size=cfg.batch_size*2, shuffle=True)

    for j, (item_id, item_features, item_text_embed, item_image_embd) in enumerate(loader_item):
        item_id, item_features, item_text_embed, item_image_embd = \
            item_id.to(cfg.device),item_features.to(cfg.device), item_text_embed.to(cfg.device), item_image_embd.to(cfg.device)

        item_embedding = model.compute_item_embeddings(
                item_id,
                item_features,
                item_text_embed,
                item_image_embd
        )
        all_item_embedding = torch.cat([all_item_embedding, item_embedding], dim=0)
        all_item_id = torch.cat([all_item_id, item_id], dim=0)

    model.save_all_item_embedding(all_item_embedding, all_item_id)

    # ...... go through the test data, and check if the searched items for the test user are among the top k list .....
    L_test_loss = []
    L_accuracy = []
    for j, (user_id, user_features, user_history, item_id, item_features, item_text_embed, item_image_embd, labels, item_log_freq) \
            in enumerate(test_loader):
        user_id, user_features, user_history, item_id, item_features, item_text_embed, item_image_embd, labels, item_log_freq = \
            user_id.to(cfg.device), user_features.to(cfg.device), user_history.to(cfg.device), \
            item_id.to(cfg.device),item_features.to(cfg.device), item_text_embed.to(cfg.device), item_image_embd.to(cfg.device),\
            labels.to(cfg.device), item_log_freq.to(cfg.device)

        loss = model.forward_train(
            user_id,  # : torch.Tensor,  # [B]
            user_features,  # : torch.Tensor,  # [B, IU]
            user_history,  # : torch.Tensor,  # [B, H]
            item_id,  # : torch.Tensor,  # [B]
            item_features,  # : torch.Tensor,  # [B, II]
            item_text_embed,
            item_image_embd,  # : torch.Tensor,  # [B, channel x H x w]
            item_log_freq,  # torch.Tensor  # [B]
        )
        L_test_loss.append(loss.item())

        top_items_inx = model.forward(
            user_id,  # [B]
            user_features,  # [B, IU]
            user_history,  # [B, H]
        )

        # if the predict item is in the top k item, then set accuracy as 1.
        L_accuracy += [(item_id[ii] in top_items_inx[ii]) for ii in range(len(top_items_inx))]

    print(f"======================================")
    model.train()
    avg_test_loss = sum(L_test_loss) / len(L_test_loss)
    writer.add_scalar("Loss/test", avg_test_loss, tb_step)
    print(f"Test loss: {avg_test_loss}")

    avg_test_accuracy = sum(L_accuracy) / len(L_accuracy)
    writer.add_scalar(f"Top {str(int(cfg.num_items))} Accuracy/test", avg_test_accuracy, tb_step)
    print(f"Top {str(int(cfg.num_items))} Test Accuracy: {avg_test_accuracy}")
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

writer.flush()
writer.close()

