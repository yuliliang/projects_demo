import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import AcuteRehabRecoverRate_Yuli__config_ as cfg
connect_EDW_by_JDB = cfg.connect_EDW_by_JDB
# @@@@@@@@@@@@@@@@@@@@ raw data visualization @@@@@@@@@@@@@@@@@
def feature_data_visualization(df, target_format):
    """
    Goal: Plot and save the figures that shows each feature vs. target value.
        This is for a quick data review.

    Args:
        df: Dataframe with
        target_format: 'classification' (categorical values) or regression (numerical values)

    Returns:
        No return values. The figures would be saved within the same folder.
    """
    # ========= for numeric feature, denote the CoN with outlier ============
    # check which CoN With possibly outlier
    df_outlier = df.describe()   # generate a table that denote outlier, only for numeric features.
    df_outlier.loc['IQR', :] = df_outlier.loc['75%', :] - df_outlier.loc['25%', :]
    df_outlier.loc['outlier_low', :] = df_outlier.loc['25%', :] - (df_outlier.loc['IQR', :] * 3)  # 1.5
    df_outlier.loc['outlier_up', :] = df_outlier.loc['75%', :] + (df_outlier.loc['IQR', :] * 3)    # 1.5
    df_outlier.loc['with_outlier', :] = (
        ((df_outlier.loc['outlier_up', :] < df_outlier.loc['max', :]) |
         (df_outlier.loc['outlier_low', :] > df_outlier.loc['min', :])).astype(int))
    df_outlier.to_csv('df_outlier.csv')

    CoN_target = cfg.CoN_target
    List_CoN = df.columns
    print('Generate data distribution plot')
    for CoN in List_CoN:
        if CoN in (cfg.CoN_ID + cfg.CoN_discard):
            continue

        NULL_percent = (df.loc[:,CoN]).isnull().sum()/df.shape[0]
        print(CoN + ': NULL %: ' + str(NULL_percent)[:5] + ')')
        if NULL_percent > cfg.missing_val_tolerance__plot:  # if too much NULL value, don't generate plots
            continue

        if df[CoN].dtype == object:   # categorical data

            fea_cardinality = int(len(df.loc[:, CoN].unique()))
            print('    :Categorical feature (' + ', cardinality = ' + str(fea_cardinality) + ')')
            if len(df.loc[:, CoN].unique()) > cfg.cat_fea_cardinality_thr:
                print("    ---- Warning: cardinality > " + str(cfg.cat_fea_cardinality_thr) + ". Did NOT generate figures for now.")
                continue

            if target_format == 'classification':
                plot_h = sns.catplot(data=df, x=CoN, hue=CoN_target, kind="count")
                plt.xticks(rotation=45)
            else:   # regression (numerical output)
                plot_h = sns.catplot(data=df, x=CoN, y=CoN_target, kind="box")
                plt.xticks(rotation=45)

            fig = plot_h.fig
            plt.tight_layout()
            fig.savefig("fig_fea_cat__" + CoN + ".png")
        else:  # numerical data
            if df_outlier.loc['with_outlier', CoN] == 1:
                print('    :Numerical feature (possible with outlier)')
            if target_format == 'classification':  # categorical

                if CoN == 'BMI':   # remove unreasonable values, and look again.
                    tmp = df[[CoN_target, CoN]]
                    tmp = tmp.loc[tmp[CoN] < cfg.input_data_range[CoN][1], :]
                    plot_h = sns.displot(data=tmp, x=CoN, hue=CoN_target, bins=200)
                    del tmp
                else:
                    plot_h = sns.displot(data=df, x=CoN, hue=CoN_target, bins=200)

            else:   # 'regression'  # numerical output s
                plot_h = sns.jointplot(data=df, x=CoN, y=CoN_target)

            fig = plot_h.fig
            plt.tight_layout()
            fig.savefig("fig_fea_num__" + CoN + ".png")

        plt.close()    # close specific sns figure: plt.close(sns_plot.fig)
    return 1

# @@@@@@@@@@@@@@@ split into train, validation, and test data @@@@@@@@@@@
def split_data_train_test(df, rand_num):
    """
    Goal: Split the full data into train / validation / test data, with an assigned random number.

    Args:
        df: Full data set.
        rand_num: The seed for random_state

    Returns:
        df_train, df_valid, df_test: Dataset for train, validation, and test.
    """
    df_train, df_valid, df_test = (
        np.split(df.sample(frac=1, random_state=rand_num),
                 [int(cfg.train_portion * len(df)), int((cfg.train_portion + cfg.valid_portion) * len(df))]))
    return df_train, df_valid, df_test

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
def data_pre_process(df_raw, train_or_test, L_process_coef):
    """
    Goal: With a input feature, map the value into a list of integers (start from 0), with None fill as -1.

    Args:
        df_raw: The data set before the pre-processing.
        train_or_test: If this is train data, the code generate processing coefficients.
            Otherwise, the codes process the data with input coefficients (i.e., L_process_coef).
        L_process_coef: A list of coefficients, which records how to pre-process the data.

    Returns:
        df: The processed input features for the model.
        df_with_ID: The processed input features, with ids, for the model.
        df_y: The processed target values for the model.
        df_y_with_ID: The processed target values, with ids, for the model.
        L_process_coef: The coefficients for performing the same pre-processing later.
    """
    df = df_raw.copy()

    df_y = df[[cfg.CoN_target]].copy()
    df_y_with_ID = (df[[cfg.CoN_target] + cfg.CoN_ID]).copy()

    for i in range(len(cfg.CoN_ID)-1, -1, -1):
        CoN = cfg.CoN_ID[i]
        tmp = df_y_with_ID.pop(CoN)
        df_y_with_ID.insert(loc=1, column=CoN, value=tmp)
    df_y_with_ID = df_y_with_ID.rename({cfg.CoN_target: cfg.CoN_target + '_Recover_home'}, axis='columns')

    # ...... data preprocessing: drop some columns & deal with missing values ......
    CoN_discard = []
    ce = cat_encode()
    if train_or_test == 'train':
        L_process_coef = {}
        N_row = df.shape[0]
        CoN_discard = cfg.CoN_discard

        # ...... denote the IQR for potentially remove outlier (before fill NULL values)...............
        df_outlier = df.describe()  # generate a table that denote outlier, only for numeric features.
        df_outlier.loc['IQR', :] = df_outlier.loc['75%', :] - df_outlier.loc['25%', :]
        df_outlier.loc['outlier_low', :] = df_outlier.loc['25%', :] - (df_outlier.loc['IQR', :] * cfg.outlier_coef)
        df_outlier.loc['outlier_up', :] = df_outlier.loc['75%', :] + (df_outlier.loc['IQR', :] * cfg.outlier_coef)

        # ...... deal with feature type & missing values first .......
        for CoN in df.columns:
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

            else:
                if df[CoN].dtype == 'object':
                    df.loc[:, CoN].fillna(cfg.object_fillna_value, inplace=True)
                    L_process_coef[CoN] = ['cat', cfg.object_fillna_value]
                else:   # numerical data
                    if cfg.Q_add_bool_col_num_fea:
                        df[cfg.flag_preStr + CoN] = df[CoN].apply(lambda x: 0 if pd.isna(x) else 1)
                        L_process_coef[cfg.flag_preStr + CoN] = ['num', '']

                    if cfg.input_data_fillna[CoN] == 'median':
                        val_4_null = (df.loc[:, CoN]).median()
                    else:
                        val_4_null = cfg.input_data_fillna[CoN]
                    df.loc[:, CoN].fillna(val_4_null, inplace=True)
                    L_process_coef[CoN] = ['num', val_4_null]

        # ...... deal with outlier (only for numerical CoN) ....
        if cfg.Q_remove_outlier:
            for CoN in L_process_coef:
                if cfg.Q_add_bool_col_num_fea:
                    if CoN[:len(cfg.flag_preStr)] == cfg.flag_preStr:  # start with '01_'   # CoN_flag = '01_' + CoN
                        continue

                if L_process_coef[CoN][0] == 'num':
                    L_process_coef[CoN].append(df_outlier.loc['outlier_low', CoN])
                    L_process_coef[CoN].append(df_outlier.loc['outlier_up', CoN])
                    df.loc[df.loc[:, CoN] < L_process_coef[CoN][2], CoN] = L_process_coef[CoN][2]  # lower bound
                    df.loc[df.loc[:, CoN] > L_process_coef[CoN][3], CoN] = L_process_coef[CoN][3]  # upper bound

        # ...... deal with categorical features encoding .....
        if cfg.Q_encode_cat_fea:
            for CoN in L_process_coef:
                if L_process_coef[CoN][0] == 'cat':
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
            elif (L_process_coef[CoN][0] == 'cat') | (L_process_coef[CoN][0] == 'num'):
                if cfg.Q_add_bool_col_num_fea:
                    if (L_process_coef[CoN][0] == 'num'):
                        df[cfg.flag_preStr + CoN] = df[CoN].apply(lambda x: 0 if pd.isna(x) else 1)
                df.loc[:, CoN].fillna(L_process_coef[CoN][1], inplace=True)
            else:
                print('Error (Yuli): Unrecognized L_process_coef[CoN][0] = ' + str(L_process_coef[CoN][0]))
                sys.exit()

        if cfg.Q_remove_outlier:
            for CoN in L_process_coef:
                if cfg.Q_add_bool_col_num_fea:
                    if CoN[:len(cfg.flag_preStr)] == cfg.flag_preStr:
                        continue

                if L_process_coef[CoN][0] == 'num':
                    df.loc[df.loc[:, CoN] < L_process_coef[CoN][2], CoN] = L_process_coef[CoN][2]  # lower bound
                    df.loc[df.loc[:, CoN] > L_process_coef[CoN][3], CoN] = L_process_coef[CoN][3]  # upper bound

        if cfg.Q_encode_cat_fea:
            for CoN in L_process_coef:
                if L_process_coef[CoN][0] == 'cat':
                    ce.map2int(df, CoN, L_process_coef[CoN][2])

    else:
        print("Error (Yuli): 'train_or_test' is not 'train' nor 'test'.")
        sys.exit()

    tmp = list(df.columns)
    for CoN in CoN_discard:
        if CoN in tmp:
            del df[CoN]
        else:
            print(CoN + " is not in the df.")

    df_with_ID = df.copy()
    for i in range(len(cfg.CoN_ID)-1, -1, -1):
        CoN = cfg.CoN_ID[i]
        tmp = df_with_ID.pop(CoN)
        df_with_ID.insert(loc=0, column=CoN, value=tmp)
    df_with_ID.insert(loc=0, column=cfg.CoN_target + '_Recover_home',
                      value=df_y_with_ID[cfg.CoN_target + '_Recover_home'])

    for CoN in cfg.CoN_ID:
        del df[CoN]

    return df, df_with_ID, df_y, df_y_with_ID, L_process_coef


# @@@@@@@@@@@@@@@@@@@@ query data @@@@@@@@@@@@@@@@@
def query_data():
    """
    Goal: Query data with SQL, or retrieve from saved files.

    Args:

    Returns:
        df: The queried data.
    """
    # ====== Instead of query, load saved data ======
    # df = pd.read_excel(cfg.in_file_name, dtype=cfg.input_data_type) # take around 20 minutes.
    # df = df.loc[:20000, :]

    # ===== connect EDW from JDB ======
    conn = connect_EDW_by_JDB()
    curs = conn.cursor()
    # myquery = "select * from visit_dm.visit where rownum<5"
    def execute_sql(q_drop, q):
        try:
            curs.execute(q_drop)  # "DROP TABLE U6055216.rehab_fea_flow_sheet_measure")
        except:
            print("Yuli: table not exist")
        curs.execute(q)

    execute_sql(cfg.query0_drop, cfg.query0)    # 2024_0814: probably take around 22 minutes
    execute_sql(cfg.query1_rehab_fea_flow_sheet_measure_drop, cfg.query1_rehab_fea_flow_sheet_measure)

    execute_sql(cfg.query2_rehab_fea_time_series_mobility_score_drop, cfg.query2_rehab_fea_time_series_mobility_score)
    execute_sql(cfg.query2_rehab_fea_time_series_PT_therapy_drop, cfg.query2_rehab_fea_time_series_PT_therapy)
    execute_sql(cfg.query2_rehab_fea_time_series_OT_therapy_drop, cfg.query2_rehab_fea_time_series_OT_therapy)
    execute_sql(cfg.query2_rehab_fea_time_series_OT_activity_score_drop, cfg.query2_rehab_fea_time_series_OT_activity_score)
    execute_sql(cfg.query2_rehab_fea_time_series_score_D_therapy_time_drop, cfg.query2_rehab_fea_time_series_score_D_therapy_time)
    execute_sql(cfg.query2_rehab_fea_Vital_for_score_date_drop, cfg.query2_rehab_fea_Vital_for_score_date)   # this is a little bit long, but less than 1 minutes for inferencing
    execute_sql(cfg.query2_rehab_fea_surgery_drop, cfg.query2_rehab_fea_surgery)
    execute_sql(cfg.query2_rehab_fea_time_series_combine_drop, cfg.query2_rehab_fea_time_series_combine)

    execute_sql(cfg.query3_rehab_fea_all_drop, cfg.query3_rehab_fea_all)
    df = pd.read_sql_query(cfg.query3_rehab_fea_all__get_data, con=conn)  # store query results

    for CoN in list(df.columns):  # make sure the data type are desired.
        if CoN in cfg.input_data_type:
            if df[CoN].dtype != cfg.input_data_type[CoN]:
                df[CoN] = df[CoN].astype(cfg.input_data_type[CoN])
    curs.close()
    conn.close()
    print("Yuli: connection close!")

    return df



