"""
< Major framework of this set of codes >
- data query (static data & time series data)
- data exploit (visualize data vs. target value)
- data split (regular split, or split based on certain column, e.g., patient_id)
- data pre-processing (null value, categorical feature coding, outlier)
- model (xgb, random forest, lightgbm; early stopping; with grid search (in it, split k-fold data with patient_id & evaluate by AUC))
- evaluation (AUC of precision recall, SHAP, feature importance)
"""
import pandas as pd
import numpy as np
from time import time
import math

from AcuteRehabRecoverRate_Yuli__data_query_ import query_data, feature_data_visualization
from AcuteRehabRecoverRate_Yuli__model_ import model_train
import AcuteRehabRecoverRate_Yuli__config_ as cfg

def split_data_train_test(df, rand_num):
    """
    Goal: Split the data into train / validation / test data by Patient_id.

    Args:
        df: Full data set.
        rand_num: The seed for random_state

    Returns:
        df_train, df_valid, df_test: Dataset for train, validation, and test.
    """
    tmp = df['PAT_ID'].value_counts(sort=False)  # get count of each patient,
    tmp = tmp.sample(frac=1, random_state=rand_num)  # permute it,
    tmp = tmp.cumsum(axis=0)

    cut_train = df.shape[0] * cfg.train_portion
    cut_valid = df.shape[0] * (cfg.train_portion + cfg.valid_portion)
    PAT_ID_train = (tmp[tmp <= cut_train]).index.to_list()
    PAT_ID_valid = (tmp[(tmp > cut_train) & (tmp <= cut_valid)]).index.to_list()
    PAT_ID_test = (tmp[tmp > cut_valid]).index.to_list()

    df_train = df[df['PAT_ID'].isin(PAT_ID_train)]
    df_valid = df[df['PAT_ID'].isin(PAT_ID_valid)]
    df_test = df[df['PAT_ID'].isin(PAT_ID_test)]

    return df_train, df_valid, df_test


if __name__ == "__main__":

    file_str = cfg.attach_file_str
    st = time()
    df = query_data()
    et = time()
    elapsed_time = time() - st
    print('Load data time:', elapsed_time/60., ' minutes')

    def map_column_val_by_input_list(df, CoN, List_val):
        """ A small function to map input list into a 0 to n value (i.e., [0 ... n]) """
        df[CoN] = df[CoN].map({val:inx for inx, val in enumerate(List_val)}).astype('object')
    map_column_val_by_input_list(df, 'RECORDED_WEEKDAY', [np.nan, 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'])
    map_column_val_by_input_list(df, 'RUCA1', [np.nan, '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])

    # ========== polish the target values into the desired format first =======
    # only generate prediction when the previous mobility score is less than 20.
    Inx_predict_group = (df[cfg.CoN_target].notnull()) & (
                (df['PRIOR_1ST_MOBILITY_SCORE'] < 20) | (df['PRIOR_1ST_MOBILITY_SCORE'].isnull()))  # 234_898

    Inx_predict_group_without_cold_start = (Inx_predict_group & (
                (df['PRIOR_1ST_MOBILITY_SCORE'].notnull()) & (df['PRE_OT_ACTIVITY_SCORE'].notnull())))  # 138_047

    Inx_predict_group_only_cold_start = (Inx_predict_group & (
                (df['PRIOR_1ST_MOBILITY_SCORE'].isnull()) | (df['PRE_OT_ACTIVITY_SCORE'].isnull())))  # 96_851

    if cfg.filtered_data == 'NonCold':  # 'ColdNonCold' # include both cold and non-cold data # 'Cold': only cold start data;
        data_filter_inx = (Inx_predict_group_without_cold_start)
    elif cfg.filtered_data == 'ColdNonCold':  # include both cold and non-cold data # 'Cold': only cold start data;
        data_filter_inx = (Inx_predict_group)
    elif cfg.filtered_data == 'Cold':  # 'ColdNonCold' # include both cold and non-cold data # 'Cold': only cold start data;
        data_filter_inx = (Inx_predict_group_only_cold_start)

    df = df.loc[data_filter_inx, :]

    df[cfg.CoN_target + '_raw'] = df[cfg.CoN_target]
    df_copy = df.copy()

    if cfg.CoN_target == 'DSCH_DISP_CODE':   # if discharge to home or not
        df[cfg.CoN_target] = (df[cfg.CoN_target].isin(cfg.CoN_target_value)).astype(int)
    elif 'FUTURE_PT_SCORE_D_DAY_' in cfg.CoN_target:  # 'FUTURE_PT_SCORE_D_DAY_FROM_TODAY' etc.
        tmp = (df[cfg.CoN_target].median())
        df[cfg.CoN_target] = (df[cfg.CoN_target] > tmp).astype(int)  # simply separate in to 2 group first.
    elif cfg.CoN_target == 'FINAL_PT_SCORE_PER_VISIT':
        tmp = (df[cfg.CoN_target ].median())   # for full data set, the median is actually 18. GREAT!
        df[cfg.CoN_target] = (df[cfg.CoN_target] >= tmp).astype(int)
    elif cfg.CoN_target == 'FUTURE_SCORE_D_DAY_0TO432':   # if for predicting mobility score improvement.
        def weight_improved_score(row):
            return row[cfg.CoN_target] * math.log2(max(22 - row['CURRENT_MOBILITY_SCORE'], 2))

        if cfg.model_target_type == 'binary_class':
            tmp = np.percentile(df[cfg.CoN_target], 50)
            df[cfg.CoN_target] = (df[cfg.CoN_target] > tmp).astype(int)

        elif cfg.model_target_type == 'regression':
            # ------ for regression, cap the extreme value in the target ------
            CoN_target_dist = df[cfg.CoN_target].describe()
            CoN_target_min = CoN_target_dist['25%'] - (CoN_target_dist['75%'] - CoN_target_dist['25%']) * 1.5
            CoN_target_max = CoN_target_dist['75%'] + (CoN_target_dist['75%'] - CoN_target_dist['25%']) * 1.5
            df.loc[(df[cfg.CoN_target] < CoN_target_min), cfg.CoN_target] = CoN_target_min
            df.loc[(df[cfg.CoN_target] > CoN_target_max), cfg.CoN_target] = CoN_target_max

    df_copy[:1000].to_csv('Model_output__raw_df' + file_str + '.csv')
    del df[cfg.CoN_target + '_raw']
    df[:1000].to_csv('Model_output__df_FinalTarget' + file_str + '.csv')

    # ===== plot data distribution, feature vs. target value =====
    if cfg.Q_examine_data:
        df_ = df.copy()
        target_format = 'classification'  # 'regression'
        tmp_v = np.percentile(df[cfg.CoN_target], 60)
        df_[cfg.CoN_target] = (df_[cfg.CoN_target] > tmp_v).astype(int)
        feature_data_visualization(df_, target_format)

    # change the BMI to reasonable range [0, 100].
    for CoN in cfg.input_data_range:
        df.loc[df[CoN] < cfg.input_data_range[CoN][0], CoN] = cfg.input_data_range[CoN][0]
        df.loc[df[CoN] > cfg.input_data_range[CoN][1], CoN] = cfg.input_data_range[CoN][1]

    # ===== Train & Testing  =====
    df_best_para = pd.DataFrame()

    df_perf = pd.DataFrame(columns=['mae', 'AUC_RecallPrecision', 'roc_auc', 'train/test time (min)'])

    for rand_num in cfg.List_rand_num:
        np.random.seed(rand_num)
        df_train, df_valid, df_test = split_data_train_test(df, rand_num)

        st = time()
        mae, AUC_RecallPrecision, roc_auc \
            = model_train(df_train, df_valid, df_test, rand_num, file_str, df_best_para)
        elapsed_time = time() - st
        print(str(rand_num) + ': Model train + test time:', elapsed_time/60., ' minutes')

        df_perf.loc[rand_num, 'mae'] = mae
        df_perf.loc[rand_num, 'AUC_RecallPrecision'] = AUC_RecallPrecision
        df_perf.loc[rand_num, 'roc_auc'] = roc_auc
        df_perf.loc[rand_num, 'train/test time (min)'] = elapsed_time / 60.

    df_perf.loc['mean', :] = df_perf.mean(axis=0)
    print(df_perf)
    df_perf.to_csv('df_perf' + file_str + '.csv')

    df_best_para.to_csv('Model_output__best_para' + file_str + '.csv')
    print("Yuli: done")