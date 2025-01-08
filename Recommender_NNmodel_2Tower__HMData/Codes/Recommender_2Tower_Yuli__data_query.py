import pandas as pd
import numpy as np
import datetime, pickle
from time import time
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import torch, sys
from Recommender_2Tower_Yuli__call_AE_model_for_img_embed import extract_img_embedding
import Recommender_2Tower_Yuli__config as cfg


# @@@@@@@@@@@@@@@@@@@@ encode categorical value in dataframe @@@@@@@@@@@@@@@@@
class cat_encode():
    """
    A class that include several function to process categorical features.
    """
    def List_val__sort(self, df, CoN):
        """
        Goal: Generate a list of column value that sort by value names.

        Args:
            df: Full data set.
            CoN: The column name of the feature we want to process.

        Returns:
            L: a list of values that sort by value names in df[CoN].
                The fill value for null will be placed as first value in L.
        """
        L = (df[CoN].unique().tolist())
        L.sort()
        if cfg.object_fillna_value in L:
            L.remove(cfg.object_fillna_value)
        L.insert(0, cfg.object_fillna_value)
        return L

    def List_val__top_count_n(self, df, CoN, top_value_coverage=0.9):
        """
        Goal: Generate a list of values & more frequent ones cover the top x % of data.

        Args:
            df: Full data set.
            CoN: The column name of the feature we want to process.
            top_value_coverage: The output values cover this proportion of data.

        Returns:
            top_cat: a list of more frequent values cover the top x % of data.
        """
        df.loc[:, CoN].fillna(cfg.object_fillna_value, inplace=True)
        tmp_df = pd.DataFrame(df[CoN].value_counts(sort=True))
        tmp_cumsum_values = np.cumsum(tmp_df['count'].values)
        top_cat_inx = (tmp_cumsum_values / tmp_cumsum_values[-1]) <= top_value_coverage
        top_cat = tmp_df.index[top_cat_inx].to_list()  #
        return top_cat

    def map2int(self, df, CoN, List_val):
        """
        Goal: With a input feature, map the value into a list of integers (start from 0), with None fill as -1.

        Args:
            df: Full data set.
            CoN: The column name of the feature we want to process.
            List_val: A list of values in this column that would be converted. Order is the assigned integer.

        Returns:
            No return values. The column value will be changed in place.
        """
        df[CoN] = df[CoN].map({val:inx for inx, val in enumerate(List_val)})
        df[CoN].fillna(-1, inplace=True)
        df[CoN] = df[CoN].astype('int')
        return 1

    def one_hot_coding(self, df, CoN, List_top_val=[]):
        """
        Goal: With a input feature, map the value into a list of integers (start from 0), with None fill as -1.

        Args:
            df: Full data set.
            CoN: The column name of the feature we want to process.
            List_top_val: A list of values in this column that would be converted into 1-hot coding.
                The values that is not in this list would be treated as none & accumulate in 1 column.
        Returns:
            No return values. The target column would be deleted and 1-hot columns attached.
        """
        if List_top_val:
            df[CoN] = df[CoN].apply(lambda x: x if (x in List_top_val) else np.nan)
        dummies = pd.get_dummies(df[CoN], prefix=CoN, prefix_sep='_', dummy_na=False)   # .astype(int)
        del df[CoN]
        df = pd.concat([df, dummies], axis=1)
        return df

# @@@@@@@@@@@@@@@@@@@@ data preprocessing @@@@@@@@@@@@@@@@@
def df_inter_add_user_history(df_inter_, df_items_):
    # ======= generate user history for each transaction ====
    # (user history depends on time, so far the interaction between user and item)

    # # ..... get 100 customer, and have a sample file to work on the user history data .....
    # # df_inter___ = df_inter_.loc[:99, :].copy()  # TODO: for quick test
    # # tmp = df_inter___[['customer_id']].drop_duplicates()
    # # tmp2 = df_inter_.merge(tmp, how='inner', on='customer_id')
    # # # generate a df_items_ with article_id, instead of article_id_inx
    # # tmp3 = tmp2.merge(df_items_, how='inner', on='article_id')
    # # tmp4 = tmp3.sort_values(by=['customer_id', 't_dat'], ascending=False)
    # # tmp4.to_csv('user_history_raw.csv')
    #
    # df_u_hist = pd.read_csv('user_history_raw.csv')
    # del df_u_hist['Unnamed: 0']
    # del df_u_hist['FName']

    df_inter_ = df_inter_[[cfg.CoN_user_id_inx, cfg.CoN_item_id_inx, 'price', 't_dat']].copy()
    df_u_hist = df_inter_.merge(df_items_, how='left', on=[cfg.CoN_item_id_inx])
    del df_inter_, df_items_

    user_id = cfg.CoN_user_id_inx   # 'customer_id' # the user_id I used in the following context.

    # input must be df_inter merge with df_items category info.
    df_u_hist['t_dat'] = pd.to_datetime(df_u_hist['t_dat'])

    item_cat_CoN = []  # the CoN in the df_items that present the category of product (using 1-hot), and could be * price later.
    for CoN in df_u_hist.columns.to_list():
        for ii in cfg.items_1hot_CoN:
            if (ii + '_') in CoN:
                item_cat_CoN.append(CoN)
                break


    # for both the user history data, sum up the category weight for each day (the date is order ascending)
    df_u_hist_PerDay = df_u_hist.groupby([user_id, 't_dat'], as_index=False)[item_cat_CoN] \
        .apply(lambda x: x.astype(float).sum()).reset_index()
    del df_u_hist

    if False:
        # ...... construct a dataframe that weight each category by price, not 1-0 ......
        df_u_hist_byPrice = df_u_hist.copy()
        for CoN in item_cat_CoN:
            df_u_hist_byPrice[CoN] *= df_u_hist_byPrice['price']

        df_u_hist_byPrice_PerDay = df_u_hist_byPrice.groupby([user_id, 't_dat'], as_index=False)[item_cat_CoN] \
            .apply(lambda x: x.astype(float).sum()).reset_index()

    # within each user, for each transaction, aggregate the previous history by weighted sum based on date. set
    # a 2 year window,  for now.
    # TODO: later could also calculate the transaction and average duration.

    # create the CoN for the user history features
    user_hist_CoN = ['hist_' + CoN for CoN in item_cat_CoN]
    # majorly the weight sum in each item category, the weight decay by date
    # user_hist_CoN += ['avg_trans_period']   #  (newest_trans_date - oldest_trans_date)/(#_of_trans -1). This is availabe when #_of_trans >= 2
    df_u_hist_PerDay[user_hist_CoN] = 0  # initialize the user_history as all zero vectors.
    # df_u_hist_PerDay['avg_trans_period'] = 0
    df_u_hist_PerDay['weight'] = 0  # for store the weight temporary.

    tmp_groups = df_u_hist_PerDay.sort_values(by='t_dat', ascending=True).groupby(user_id, as_index=False)

    def get_user_hist(df_g):
        # df_g is group value, the dataframe

        # def generate_user_history(df_g, item_cat_CoN, user_hist_CoN):
        df_g_idx = df_g.index.tolist()  # smaller index = older t_dat
        for i in range(0, len(df_g_idx) - 1):
            idx_hist = df_g_idx[i]
            idx_later = df_g_idx[i + 1:]

            # ..... calculate the weight of current row as historical data to the later row:
            df_g.loc[idx_later, 'weight'] = \
                (730 - ((df_g.loc[idx_later, 't_dat'] - (df_g.loc[idx_hist, 't_dat'])).dt.days)) / (730)
            df_g.loc[idx_later, 'weight'] = df_g.loc[idx_later, 'weight'].clip(lower=0)  # make sure the weight >= 0
            # or, currently, the history older than 2 years (365 * 2 = 730 days) won't count into user history

            # ..... add category weight of current item, the oldest item, to the later user_history ....
            # (1) make the weight with same dimension like category feature from (d_dim,) to (d_dim x 1)
            # (2) make the current row product category weight from (fea_dim) to (1 x fea_dim)
            # (3) make it become (d_dim x 1) @ (1 x fea_dim) = d_dim x fea_dim, to be add back to the user_history of later date.
            #      "np.dot" or "@" is matrix multiplication
            # (4) add to the user history categorical item in the later rows.
            df_g.loc[idx_later, user_hist_CoN] += \
                (np.array(df_g.loc[idx_later, 'weight'])[:, np.newaxis]) @ \
                (np.array(df_g.loc[idx_hist, item_cat_CoN])[np.newaxis, :])
                # (np.array(df_g.loc[idx_later, 'weight'])[:, np.newaxis] @ np.ones((1, len(item_cat_CoN)))) * \
                # np.array(df_g.loc[idx_later, item_cat_CoN])

        return df_g

    df_u_hist_PerDay_withUserHist = tmp_groups.apply(get_user_hist)  # after this, it might be a tuple, not a dataframe.
    del df_u_hist_PerDay_withUserHist['weight']

    df_u_hist_PerDay_withUserHist = df_u_hist_PerDay_withUserHist.reset_index()   # after this, it might be a tuple, not a dataframe.
    df_u_hist_PerDay_withUserHist = df_u_hist_PerDay_withUserHist[[user_id, 't_dat'] + user_hist_CoN]

    return df_u_hist_PerDay_withUserHist, user_hist_CoN


def load_and_process_data():   # need to differentiate train & test.
    ce = cat_encode()

    # ====== process the raw data set and into a smaller dataset ======
    # df_items = pd.read_csv(cfg.in_file_dir + "articles.csv")
    # # product detail description: articles['detail_desc'] <---- could further be processed as embedding.
    # # # get the related image using articles['article_id']
    # # img = mpimg.imread(f'../input/h-and-m-personalized-fashion-recommendations/images/0{str(data.article_id)[:2]}/0{int(data.article_id)}.jpg')
    # # ==> good..... so just put the basic number in --> 2 tower --> training process visulization, parameter tuning --> meta data. YES!
    #
    # df_users = pd.read_csv(cfg.in_file_dir + "customers.csv")
    # df_inter = pd.read_csv(cfg.in_file_dir + "transactions_train.csv")
    # # sample_submission = pd.read_csv(cfg.in_file_dir + "sample_submission.csv")
    #
    # # ........ limit the top 3_000 popular items based on import image file path .......
    # df_img = pd.read_excel(cfg.img_path_file)   # the top 3_000 popular items with images.
    #
    # df_inter_ = df_inter.merge(df_img[['article_id', 'FName']], how="inner", on="article_id")  # ~ 30_000_000 --> ~ 10_465_963
    # del df_inter
    # df_inter_.reset_index(drop=True, inplace=True)
    #
    # df_items = df_items.merge(df_img[['article_id']], how='inner', on='article_id')  # 105_542 --> 2_991
    # df_items.reset_index(drop=True, inplace=True)
    #
    # # to make the user history be feasible to process in the laptop, keep only top 10_000 users from interaction table.
    # select_user = df_inter_['customer_id'].value_counts().index.tolist()[:cfg.keepTopNUsers]
    # df_inter_ = df_inter_.loc[df_inter_['customer_id'].isin(select_user), :]   # 914_354, around 1 million
    # df_users = df_users.loc[df_users['customer_id'].isin(select_user), :]   # keep 10_000 users.
    #
    # # users_keep = df_inter_[['customer_id']].drop_duplicates()
    # # df_users = df_users.merge(users_keep, how='inner', on='customer_id')  # --> 1_371_980 --> 1_067_976
    #
    # # ..... save a smaller set .....
    # pickle.dump([df_inter_, df_items, df_users],
    #             open(cfg.in_file_dir + "SmallerData_3000Items_10000Users.bin", "wb"))

    # ...... load the smaller data set ......
    tmp = pickle.load(open(cfg.in_file_dir + "SmallerData_3000Items_10000Users.bin", "rb"))  # load data set before process
    df_inter_ = tmp[0]
    df_items = tmp[1]
    df_users = tmp[2]
    del tmp

    # ===========
    if False:
        df_inter_ = df_inter_.iloc[:1000, :].copy()   # TODO: for quick test
        df_inter_.reset_index(drop=True, inplace=True)

    # ....... get the 3000 images that are most popular items in the article.csv ......
    if False:   # output this file when we need to extract img file of the top used item.
        df_items['photo_dir'] = df_items['article_id'].apply(lambda x: '0' + str(x)[:2] + '/0' + str(int(x)) + '.jpg')    # extract photo path and file names.
        tmp_items = df_items[['article_id', 'photo_dir']].copy();  tmp_inter = df_inter[['article_id', 'price']].copy()
        tt = tmp_inter.merge(tmp_items, how='left', on='article_id')
        tt['photo_dir'].value_counts()[:3000].to_csv('top3000_product_photo_and_interaction_count.csv')
    # the top 3000 products already catch ~10_000_000 transaction. Use this as a sample file.

    def process_id(df_, CoN_id):
        # replace the item id with the integers start from 1
        df_.index = np.arange(1, len(df_) + 1)  # make the index start from 1 (0 is left for unseen item, or padding_idx=0 in nn.Embedding.
        df_only_id = df_[[CoN_id]].copy()
        dict_inx2id = df_only_id[CoN_id].to_dict()   # convert a series & index to a dictionary.
        df_only_id[CoN_id + '_inx'] = df_only_id.index
        df_[CoN_id + '_inx'] = df_.index
        df_[CoN_id + '_inx'] = df_[CoN_id + '_inx'].astype('int64')   # might unnecessary. But just in case.
        # del df_[CoN_id]
        return df_, df_only_id, dict_inx2id

    # ====== df_items ======
    df_items, df_items_only_id, dict_items_inx2id = process_id(df_items, 'article_id')

    # ........ extract some columns .....
    df_items_ = df_items[['article_id_inx'] + cfg.items_1hot_CoN].copy()
    for CoN in cfg.items_1hot_CoN:
        List_top_val = ce.List_val__top_count_n(df_items_, CoN, top_value_coverage=0.95)
        df_items_ = ce.one_hot_coding(df_items_, CoN, List_top_val)

    # ====== df_users ======
    df_users, df_users_only_id, dict_users_inx2id = process_id(df_users, 'customer_id')

    # .............
    df_users_ = df_users[['customer_id_inx'] + ['FN', 'Active', 'club_member_status', 'fashion_news_frequency', 'age']].copy()
    for CoN in ['FN', 'Active']:   # column values: only Nan and 1
        df_users_[CoN].fillna(0., inplace=True)

    for CoN in ['club_member_status', 'fashion_news_frequency']:  # with more than 2 values in the column
        List_top_val = ce.List_val__top_count_n(df_users_, CoN, top_value_coverage=1.0)
        df_users_ = ce.one_hot_coding(df_users_, CoN, List_top_val)

    # ..... process the age column .......
    df_users_tmp = df_users_[['age']].copy()
    bins = cfg.user_age_bins   # [15, 23, 40, 50, 100]   # bin the age value.
    bin_lower_bound = bins[:-1]
    bin_interval = [bins[i+1] - bins[i] for i in range(0, len(bins)-1)]
    df_users_tmp['age_bin_lower_bound'] = pd.cut(df_users_tmp['age'], bins=bins, labels=bin_lower_bound).astype(float)
    df_users_tmp['age_bin_interval'] = pd.cut(df_users_tmp['age'], bins=bins, labels=bin_interval).astype(float)
    df_users_tmp['age_norm'] = (df_users_tmp['age'] - df_users_tmp['age_bin_lower_bound'])/df_users_tmp['age_bin_interval']

    df_users_tmp_2 = df_users_tmp[['age_bin_lower_bound', 'age_norm']].copy()
    List_top_val = ce.List_val__top_count_n(df_users_tmp_2, 'age_bin_lower_bound', top_value_coverage=1.0)
    df_users_tmp_2 = ce.one_hot_coding(df_users_tmp_2, 'age_bin_lower_bound', List_top_val)

    List_CoN_age_1hot = (df_users_tmp_2.columns.to_list())
    df_users_tmp_2['age_norm'].fillna(1.0, inplace=True)    # fill Nan with 1, in order to multiple and keep the missing value column be the same
    List_CoN_age_1hot.remove('age_norm')
    for CoN in List_CoN_age_1hot:
        df_users_tmp_2[CoN] = df_users_tmp_2[CoN] * df_users_tmp_2['age_norm']
        df_users_tmp_2.loc[df_users_tmp_2[CoN]>1.0, CoN] = 1.0   # cap to make sure values between 1 and 0. But this might not that necessary.
        df_users_tmp_2.loc[df_users_tmp_2[CoN]<0.0, CoN] = 0.0

    df_users_ = pd.concat([df_users_, df_users_tmp_2[List_CoN_age_1hot]], axis=1)

    df_users_['age'] = df_users_['age']/100.  # assume max age is 100. df_users['age'].max()  # besides binned age, still keep the age, but normalize it.
    df_users_['age'].fillna(0., inplace=True)  # set to zero for age = np.nan, since the minimum of age is teenager.
    # print(df_users)

    # ======= df_interactions =========
    df_inter_ = df_inter_[['customer_id', 'article_id', 'price', 'FName', 't_dat']].copy()   # t_dat & sales_channel_id should pretty useful. Could think about how to use it later.
    df_inter_ = df_inter_.merge(df_items_only_id, how='left', on='article_id')
    df_inter_ = df_inter_.merge(df_users_only_id, how='left', on='customer_id')
    df_inter_['article_id_inx'] = df_inter_['article_id_inx'].astype('int64')
    df_inter_['customer_id_inx'] = df_inter_['customer_id_inx'].astype('int64') # id is int64 + all input features are float64

    st = time()
    df_user_history, user_hist_CoN = df_inter_add_user_history(df_inter_, df_items_)   # user_history, this 1 row per user per day, not multiple row per day.
    print('generating df_user_history took time (seconds): ')
    print(time()-st)   # for 3000 product & 10_000 users, 7597.248556613922  = 2.1103468212816447 hours <—— Yuli: not bad!

    # df_inter_['price'] = 1.  # let's use binary first to train, see if it's workable. later could use regression
    df_inter_['with_price'] = 1.  # let's use binary first to train, see if it's workable. later could use regression
    # tmp_mean = df_inter_['price'].mean()
    # tmp_std = df_inter_['price'].std()
    # df_inter_['price'] = (df_inter_['price'] - tmp_mean)/tmp_std   # normalized it for input.

    # use the id_index, instead of id
    # df_users_.drop(['customer_id'])
    # df_items_.drop(['article_id'])

    assert df_items_.isnull().sum().sum() == 0, 'Yuli: There is np.nan in the pre-processed df_items_.'
    assert df_users_.isnull().sum().sum() == 0, 'Yuli: There is np.nan in the pre-processed df_users_.'
    assert df_inter_.isnull().sum().sum() == 0, 'Yuli: There is np.nan in the pre-processed df_inter_.'

    pickle.dump([df_items_, df_users_, df_user_history, user_hist_CoN, df_inter_, dict_items_inx2id, dict_users_inx2id],
                open(cfg.in_file_dir + "SmallerData_3000Items_10000Users_withUserHistory.bin", "wb"))

    df_inter_.drop(columns=['article_id', 'customer_id'], inplace=True)
    pickle.dump([df_items_, df_users_, df_user_history, user_hist_CoN, df_inter_, dict_items_inx2id, dict_users_inx2id],
                open(cfg.in_file_dir + "SmallerData_3000Items_10000Users_withUserHistory__keep_only_id_inx.bin", "wb"))
    # tmp = pickle.load(
    #     open(cfg.in_file_dir + "SmallerData_3000Items_10000Users_withUserHistory__keep_only_id_inx.bin", "rb"))
    return df_items_, df_users_, df_user_history, user_hist_CoN, df_inter_, dict_items_inx2id, dict_users_inx2id


def further_process_data():
    # further_process_data(df_items_, df_users_, df_user_history, user_hist_CoN, df_inter_,
    #                      dict_items_inx2id, dict_users_inx2id)   # when there is no pre-saved data.
    # ------ load the data processed in "load_and_process_data" ------
    tmp = pickle.load(
        open(cfg.in_file_dir + "SmallerData_3000Items_10000Users_withUserHistory__keep_only_id_inx.bin", "rb"))
    df_items_ = tmp[0]
    df_users_ = tmp[1]
    df_user_history = tmp[2]
    user_hist_CoN = tmp[3]
    df_inter_ = tmp[4]
    dict_items_inx2id = tmp[5]
    dict_users_inx2id = tmp[6]
    # df_inter_ = df_inter_.iloc[:1000, :]
    # ----- temporary -----
    df_inter_['price'] = 1.

    # ----- further process user history vector ------
    # ...... get the list of CoN for each type of category columns .....
    L_CoN_per_cat_key_word = {}
    for CoN in df_user_history.columns.tolist():
        for cat_key_word in cfg.items_1hot_CoN:
            if cat_key_word in CoN:
                if cat_key_word not in L_CoN_per_cat_key_word.keys():
                    L_CoN_per_cat_key_word[cat_key_word] = [CoN]
                else:
                    L_CoN_per_cat_key_word[cat_key_word].append(CoN)

    assert sum([len(val) for key, val in L_CoN_per_cat_key_word.items()]) == len(user_hist_CoN), \
        'user_history features: sum of # of CoN in sub-category is not equal to total # of CoN.'

    # ...... have a sum up vector, and normalize the category vector .....
    df_user_history['all_cat_sum'] = 0
    for key, L_CoN in L_CoN_per_cat_key_word.items():
        df_user_history[key + '_sum'] = df_user_history[L_CoN].sum(axis=1)   # sum to a column
        df_user_history['all_cat_sum'] += df_user_history[key + '_sum']

        # normalize the vector.
        apply_inx = (df_user_history[key + '_sum'] != 0)   # make sure not divide by zero
        for CoN in L_CoN:
            df_user_history.loc[apply_inx, CoN] /= df_user_history.loc[apply_inx, key + '_sum']

        # after normalize the category vector, normalize the category sum
        df_user_history[key + '_sum'] /= (df_user_history[key + '_sum'].quantile(0.98))

    df_user_history['all_cat_sum'] /= df_user_history['all_cat_sum'].quantile(0.98)  # normalize the over all sum vector

    # -------- extract item embedding from images & desc_details -------
    dict_items_id2inx = {val: key for key, val in dict_items_inx2id.items()}  # get dict that convert item id to index

    # ........ transfer the text of clothes description into embeddings ......
    if True:   # use this later
        model_SentTrans = SentenceTransformer('all-MiniLM-L6-v2')   # each encode embedding, dimension: [384]

        df_items = pd.read_csv(cfg.in_file_dir + "articles.csv")  # get raw item data for desc_detail column
        df_items['article_id_inx'] = df_items['article_id'].apply\
            (lambda x: int(dict_items_id2inx[x]) if x in dict_items_id2inx.keys() else np.nan)  # convert id to index # the dict only contain the selected items.
        df_item_text = df_items.loc[df_items['article_id_inx'].notnull(), ['article_id_inx', 'detail_desc']]   # get product description of select items
        # list_unique_desc = df_items_['detail_desc'].unique().tolist()

        # df_item_text['detail_desc_embed'] = model_SentTrans.encode(df_item_text['detail_desc'].tolist())
        # df_item_text['article_id_inx', 'detail_desc_embed']

        dict_item_text_embed = {}
        for ii in df_item_text.index.tolist():
            item_id_inx = df_item_text.loc[ii, 'article_id_inx']   #
            desc = df_item_text.loc[ii, 'detail_desc']
            if not pd.isna(desc):  # if desc is not np.nan, pd.isna support number * object
                try:
                    dict_item_text_embed[item_id_inx] = torch.tensor(model_SentTrans.encode(desc))  # convert to tensor first.
                except Exception as e:
                    print("Error when extracting sentence embedding: " + str(e))
                    print(desc)
                    print(item_id_inx)

        # if np.nan in list_unique_desc:
        #     list_unique_desc.remove(np.nan)
        # for i in list_unique_desc:   # could also put as a list, and encode could output all embedding at one time. https://sbert.net/#usage
        #     try:
        #         dict_item_text_embed[i] = torch.tensor(model_SentTrans.encode(i))
        #     except Exception as e:
        #         print("Error when extracting sentence embedding: " + str(e))
        #         print(i)

    pickle.dump([df_items_, df_users_, df_user_history, user_hist_CoN, df_inter_, dict_items_inx2id, dict_users_inx2id
                    , dict_item_text_embed],
                open(cfg.in_file_dir + "SmallerData_3000Items_10000Users__after_further_process_data.bin", "wb"))

    return df_items_, df_users_, df_user_history, user_hist_CoN, df_inter_, dict_items_inx2id, dict_users_inx2id \
                    , dict_item_text_embed


def further_process_data_img(dict_items_inx2id):
    '''
    For 2991 image, the results matrix is equal to more than 13GB, while the model is only 266.7 MB. So choose to put
    this function a sides, and extract the embedding when loading.
    '''
    dict_items_id2inx = {val: key for key, val in dict_items_inx2id.items()}  # get dict that convert item id to index

    # # ...... extract image model, for later usage ......
    # PATH_model_early_stop = cfg.model_path + cfg.model_name  # cfg.DIR_out + 'model/' + 'model' + str_out
    # img_model, optimizer, scheduler_lr, L_norm_coef, CoN_target, epoch_min, PARA = \
    #     get_pre_exist_model(PATH_model_early_stop)
    #
    # em = EvalModel(device=cfg.device)  # set the object to evaluate model
    # embeddings, embedding_FName, List_article_id = em.create_embedding(img_model, full_loader, cfg.device)
    #
    # # update the dimension of image embedding.
    # embeddings = embeddings.reshape(embeddings.shape[0], -1)


    # .......... extract all the imaging embedding for the later usage .....
    if True:  # too big, it turns out I process in stream, instead of ahead.
        # This function also save the correct item_img_embedding_size in the function "extract_embedding"
        L_img_embed, L_img_article_id = extract_img_embedding(cfg.img_path_file)  # img embedding is output from encoder, already a tensor.
        L_img_embed = L_img_embed[1:]  # remove the first space padding one. # TODO: could think about how to remove this later
        L_img_article_id = L_img_article_id[1:]  # remove the first space padding one. # TODO: could think about how to remove this later
        L_img_item_id_inx = [dict_items_id2inx[id] for id in L_img_article_id if id in dict_items_id2inx.keys()]

        # dict_item_img_embed = {}   # TODO: probably change to dictionary later?
        # for i in range(len(L_img_item_id_inx)):
        #     dict_item_img_embed[L_img_item_id_inx[i]] = L_img_embed[i]
        # #
        # # pickle.dump(L_img_embed, L_img_item_id_inx,
        # #         open(cfg.in_file_dir + "SmallerData_3000Items_10000Users__after_further_process_data_img.bin", "wb"))

    # sys.getsizeof(my_list)
    return L_img_embed, L_img_item_id_inx   # dict_item_img_embed


# # plot the photo with detail_desc
# f, ax = plt.subplots(1, 5, figsize=(20,10))
# i = 0
# for _, data in max_price_ids.iterrows():
#     desc = articles[articles['article_id'] == data['article_id']]['detail_desc'].iloc[0]
#     desc_list = desc.split(' ')
#     for j, elem in enumerate(desc_list):
#         if j > 0 and j % 5 == 0:
#             desc_list[j] = desc_list[j] + '\n'
#     desc = ' '.join(desc_list)
#     img = mpimg.imread(f'../input/h-and-m-personalized-fashion-recommendations/images/0{str(data.article_id)[:2]}/0{int(data.article_id)}.jpg')
#     ax[i].imshow(img)
#     ax[i].set_title(f'price: {data.price:.2f}')
#     ax[i].set_xticks([], [])
#     ax[i].set_yticks([], [])
#     ax[i].grid(False)
#     ax[i].set_xlabel(desc, fontsize=10)
#     i += 1
# plt.show()

def data_pre_process(df_raw, train_or_test, L_process_coef):
    df = df_raw.copy()

    """
    Goal: data preprocessing (deal with missing value, remove outlier, convert categorical data with one-hot or
        label encoding. and save all these into a dictionary that could transfer those from train data to test data.
    :param df:
    :param train_or_test:
    :param L_process_coef:
    :return:
    """
    # ....... normalize values, if necessary .........
    # num_features = X_train.select_dtypes(include=['float']).columns
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # X_train[num_features] = scaler.fit_transform(X_train[num_features])
    # X_test[num_features] = scaler.fit_transform(X_test[num_features])
    # X_valid[num_features] = scaler.fit_transform(X_valid[num_features])

    # def normalize_num(df):          # Normalize numerical features
    #     from sklearn.preprocessing import StandardScaler
    #     num_features = df.select_dtypes(include=['float']).columns
    #     # cat_features = X_train.select_dtypes(include=['category']).columns
    #     scaler = StandardScaler()
    #     df[num_features] = scaler.fit_transform(df[num_features])
    #     # normalized_df = pd.concat([df[num_features], df[cat_features]], axis=1)
    #     return df
    # normalize_num(df)

    # ..... get target values .....
    df_y = df[[cfg.CoN_target]].copy()
    df_y_with_ID = (df[[cfg.CoN_target] + cfg.CoN_ID]).copy()  # if don't use copy, the dtype might tight with the df data taype.

    for i in range(len(cfg.CoN_ID)-1, -1, -1):
        CoN = cfg.CoN_ID[i]
        tmp = df_y_with_ID.pop(CoN)
        df_y_with_ID.insert(loc=1, column=CoN, value=tmp)
    df_y_with_ID = df_y_with_ID.rename({cfg.CoN_target: cfg.CoN_target + '_Recover_home'}, axis='columns')

    # ...... data preprocessing: drop some columns & deal with missing values ......
    CoN_discard = []
    # le = LabelEncoder()
    ce = cat_encode()
    if train_or_test == 'train':
        L_process_coef = {}
        N_row = df.shape[0]  # number of rows
        CoN_discard = cfg.CoN_discard

        # ...... denote the IQR for potentially remove outlier (before fill NULL values)...............
        df_outlier = df.describe()  # generate a table that denote outlier, only for numeric features.
        df_outlier.loc['IQR', :] = df_outlier.loc['75%', :] - df_outlier.loc['25%', :]
        df_outlier.loc['outlier_low', :] = df_outlier.loc['25%', :] - (df_outlier.loc['IQR', :] * cfg.outlier_coef)# 1.5
        df_outlier.loc['outlier_up', :] = df_outlier.loc['75%', :] + (df_outlier.loc['IQR', :] * cfg.outlier_coef) # 1.5
        # df_outlier.loc['with_outlier', :] = (((df_outlier.loc['outlier_up', :] < df_outlier.loc['max', :]) |
        #                                    (df_outlier.loc['outlier_low', :] > df_outlier.loc['min', :])).astype(int))

        # ....... remove the column if there is only 1 unique value in this column .........

        # ...... deal with feature type & missing values first .......
        for CoN in df.columns:
            # print(CoN)
            if CoN in cfg.CoN_ID:
                L_process_coef[CoN] = ['id']
                continue
            elif CoN in CoN_discard:
                L_process_coef[CoN] = ['del']
                continue
            elif (df[CoN].unique()).shape[0] == 1:  # only 1 value in this column, delete it.
                print('Yuli: Column ' + CoN + ' contain only 1 unique value in train data. Delete this column.')
                CoN_discard.append(CoN)
                L_process_coef[CoN] = ['del']
                continue
            elif (df.loc[:,CoN]).isnull().sum() / N_row > cfg.missing_val_tolerance__feature:  # too many missing values
                print('Yuli: Column ' + CoN + ' contain too many missing values. Delete this column.')
                CoN_discard.append(CoN)
                L_process_coef[CoN] = ['del']
                continue
            # elif (df.loc[:,CoN]).isnull().sum() == 0:    # no missing value
            #     if df[CoN].dtype == 'object':
            #         L_process_coef[CoN] = ['cat', '']
            #     else:   # numerical values
            #         L_process_coef[CoN] = ['num', '']  # df_outlier.loc['outlier_low', CoN], df_outlier.loc['outlier_up', CoN]]

            else:  # have some NULL or no NULL (there might be NULL in test data), then deal with missing value
                if df[CoN].dtype == 'object':  # 'object', 'O'    # label the missing value, instead of MODE here.
                    df.loc[:, CoN].fillna(cfg.object_fillna_value, inplace=True)
                    L_process_coef[CoN] = ['cat', cfg.object_fillna_value]
                else:   # numerical data
                    if cfg.Q_add_bool_col_num_fea:   # add a column to flag missing values (1: with value, 0: NULL)
                        # def flag_1(x):
                        #     return (0 if (pd.isna(x)) else 1)
                        # df[cfg.flag_preStr + CoN] = df[CoN].apply(flag_1)
                        df[cfg.flag_preStr + CoN] = df[CoN].apply(lambda x: 0 if pd.isna(x) else 1)
                        L_process_coef[cfg.flag_preStr + CoN] = ['num', '']   # the non indicator don't need to fillna.

                    if cfg.input_data_fillna[CoN] == 'median':
                        val_4_null = (df.loc[:, CoN]).median()
                    else:   # have denoted filled value for this numeric columns
                        val_4_null = cfg.input_data_fillna[CoN]
                    df.loc[:, CoN].fillna(val_4_null, inplace=True)
                    L_process_coef[CoN] = ['num', val_4_null]

        # ...... deal with outlier (only for numerical CoN) ....
        if cfg.Q_remove_outlier:
            for CoN in L_process_coef:
                # the flag column won't have outlier problem, or with L_process_coef[CoN][1] == ''
                if cfg.Q_add_bool_col_num_fea:
                    if CoN[:len(cfg.flag_preStr)] == cfg.flag_preStr:  # start with '01_'   # CoN_flag = '01_' + CoN
                        continue

                if L_process_coef[CoN][0] == 'num':
                    L_process_coef[CoN].append(df_outlier.loc['outlier_low', CoN])
                    L_process_coef[CoN].append(df_outlier.loc['outlier_up', CoN])
                    df.loc[df.loc[:, CoN] < L_process_coef[CoN][2], CoN] = L_process_coef[CoN][2]  # lower bound
                    df.loc[df.loc[:, CoN] > L_process_coef[CoN][3], CoN] = L_process_coef[CoN][3]  # upper bound

        # ...... deal with categorical features encoding .....
        if cfg.Q_encode_cat_fea:   # need to generate encoding after fill null values (Random Forest could take cateogrical features directly)
            for CoN in L_process_coef:
                if L_process_coef[CoN][0] == 'cat':
                    # df[CoN] = le.fit_transform(df.loc[:, CoN])   # generate the mapping for label encoding.
                    # L_process_coef[CoN].append(le.classes_)
                    List_val = ce.List_val__sort(df, CoN)  # get the cat value want to encode, order matter
                    ce.map2int(df, CoN, List_val)
                    L_process_coef[CoN].append(List_val)

    elif train_or_test == 'test':
        # L_process_coef[Con] format:
        #   CoN are id, like patient_id, visit_no: ['id']
        #   CoN are not used as feature: ['del']
        #   CoN is categorical feature: ['cat', value_for_null, categorical_feature_mapping]
        #   CoN is numerical features: ['num', value_for_null, outlier_lower_bound, outlier_upper_bound]
        # only when CoN is additional nan indicator column will have special format: ['num', '']
        CoN_discard = []
        for CoN in L_process_coef:
            if L_process_coef[CoN][0] == 'id':
                continue
            elif L_process_coef[CoN][0] == 'del':
                CoN_discard.append(CoN)
                continue
            elif (L_process_coef[CoN][0] == 'cat') | (L_process_coef[CoN][0] == 'num'):   # using filled value in training data
                if cfg.Q_add_bool_col_num_fea:  # add a column to flag missing values (1: with value, 0: NULL)
                    if (L_process_coef[CoN][0] == 'num'):   # numerical data
                        # def flag_2(x):
                        #     return (0 if (pd.isna(x)) else 1)
                        # df[cfg.flag_preStr + CoN] = df[CoN].apply(flag_2)
                        df[cfg.flag_preStr + CoN] = df[CoN].apply(lambda x: 0 if pd.isna(x) else 1)
                df.loc[:, CoN].fillna(L_process_coef[CoN][1], inplace=True)
            else:
                print('Error (Yuli): Unrecognized L_process_coef[CoN][0] = ' + str(L_process_coef[CoN][0]))
                sys.exit()

        if cfg.Q_remove_outlier:
            for CoN in L_process_coef:
                # the flag column won't have outlier problem, or with L_process_coef[CoN][1] == ''
                if cfg.Q_add_bool_col_num_fea:
                    if CoN[:len(cfg.flag_preStr)] == cfg.flag_preStr:  # start with '01_'   # CoN_flag = '01_' + CoN
                        continue

                if L_process_coef[CoN][0] == 'num':
                    df.loc[df.loc[:, CoN] < L_process_coef[CoN][2], CoN] = L_process_coef[CoN][2]  # lower bound
                    df.loc[df.loc[:, CoN] > L_process_coef[CoN][3], CoN] = L_process_coef[CoN][3]  # upper bound

        if cfg.Q_encode_cat_fea:   # need to generate encoding after fill null values (Random Forest could take cateogrical features directly)
            for CoN in L_process_coef:
                if L_process_coef[CoN][0] == 'cat':
                    ce.map2int(df, CoN, L_process_coef[CoN][2])   #  List_val = L_process_coef[CoN][2]
                    # le.classes_ = L_process_coef[CoN][2]
                    # df[CoN] = le.transform(df.loc[:, CoN]) # testing phase, we only need to transfer, no fit
                    # # if df.loc[:, CoN] = le.fit_transform(df.loc[:, CoN])
                    # # , still stay "object", not int. later lightgbm won't take, even later use astype('category')

    else:
        print("Error (Yuli): 'train_or_test' is not 'train' nor 'test'.")
        sys.exit()

    tmp = list(df.columns)
    for CoN in CoN_discard:
        if CoN in tmp:
            del df[CoN]
        else:
            print(CoN + " is not in the df.")
            # df.drop(CoN_discard, axis=1, inplace=True)

    df_with_ID = df.copy()
    for i in range(len(cfg.CoN_ID)-1, -1, -1):
        CoN = cfg.CoN_ID[i]
        tmp = df_with_ID.pop(CoN)
        df_with_ID.insert(loc=0, column=CoN, value=tmp)
    df_with_ID.insert(loc=0, column=cfg.CoN_target + '_Recover_home',
                      value=df_y_with_ID[cfg.CoN_target + '_Recover_home'])

    # df.drop(cfg.CoN_ID, axis=1, inplace=True)
    for CoN in cfg.CoN_ID:
        del df[CoN]

    # ...... data preprocessing: categorical columns ......
    # df = one_hot_coding(df)
    # df = LabelEncord_categorical(df)

    # # convert into categorical feature for lightgbm
    # for CoN in L_process_coef:
    #     if L_process_coef[CoN][0] == 'cat':
    #         df.loc[:, CoN] = df.loc[:, CoN].astype('category')

    return df, df_with_ID, df_y, df_y_with_ID, L_process_coef


# @@@@@@@@@@@@@@@@@@@@ query data @@@@@@@@@@@@@@@@@
def query_data():
    df = pd.read_excel(cfg.in_file_name, dtype=cfg.input_data_type)  # dtype={'RECORDED_DATE':str, 'MAX_SCORE': float}
    return df



