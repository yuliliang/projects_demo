from time import time
import pickle, shap, math, pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Skopt functions
from skopt import BayesSearchCV
import AcuteRehabRecoverRate_Yuli__config_ as cfg
from AcuteRehabRecoverRate_Yuli__data_query_ import data_pre_process
from bayes_opt import BayesianOptimization
from sklearn.metrics import (accuracy_score, precision_recall_curve, mean_absolute_error
        , average_precision_score, f1_score, roc_curve, roc_auc_score, make_scorer)

if cfg.model_type == 'xgb':
    import xgboost as xgb
    from xgboost import plot_importance
elif cfg.model_type == 'lightgbm':
    import lightgbm as lgb
elif cfg.model_type == 'rf':
    from sklearn.ensemble import RandomForestClassifier

# @@@@@@@@@@@@@@@ use logistic regression to find a boundary from a group of numbers with binary labels  @@@@@@@@@@@
def find_threshold_through_logistic_regression(X, y):
    """
    Goal: With intput data and a binary label, find a x value that split the data by logistic regression.

    Args:
        X: Input data
        y: The target value

    Returns:
        cut_off_number: a value found by logistic regress that split the data in two.
    """
    clf = LogisticRegression()
    clf.fit(X, y)
    cut_off_number = (-1 * clf.intercept_[0] / clf.coef_[0])[0]
    del clf
    return cut_off_number


def claim_model_obj(para):
    """
    Goal: Claim the correct model object.
    """
    if cfg.model_type == 'xgb':
        return xgb.XGBClassifier(**para)  # Train data using XGBRegressor with the best parameter
    elif cfg.model_type == 'lightgbm':
        if cfg.model_target_type == 'regression':
            return lgb.LGBMRegressor(**para)
        else:
            return lgb.LGBMClassifier(**para)
    elif cfg.model_type == 'rf':
        return RandomForestClassifier(**para)


def visualize_TP_FP_TN_FN_per_month(X_test_with_ID, rand_num, file_str):
    """
    Goal: Generate a figure that visualized the TP, FP, TN, FN proportion for each month.
    """
    tmp = X_test_with_ID[[cfg.CoN_target + '_Recover_home', 'Prediction', 'PAT_ID', 'RECORDED_DATE']].copy()
    tmp['year'] = tmp['RECORDED_DATE'].dt.year
    tmp['month'] = tmp['RECORDED_DATE'].dt.month
    tmp = tmp.loc[tmp['year'] == 2023, :]  # focus on the 2023 data.
    tmp = tmp.sample(frac=1, random_state=rand_num)
    tmp = tmp[[cfg.CoN_target + '_Recover_home', 'Prediction', 'PAT_ID', 'month']].copy()
    tmp = tmp.groupby(['month', 'PAT_ID']).head(1).reset_index(drop=True)
    tmp['Prediction_by_70_per'] = 0

    #...... make the top 70% prediction for each day are positive ....
    def top_70_percent_group(df_group):
        top_70_per_count = int(cfg.Rehab_staff_cover * len(df_group))
        top_70_per_inx = df_group.sort_values(by='Prediction', ascending=False).iloc[:top_70_per_count].index  # top 70 index
        df_group.loc[top_70_per_inx, 'Prediction_by_70_per'] = 1
        return df_group

    result_df = tmp.groupby('month').apply(top_70_percent_group)

    def TP_TN_FP_FN(row):
        if row[cfg.CoN_target + '_Recover_home'] == 1:
            if row['Prediction_by_70_per'] == 1:
                return 'TP'
            else:  # row['thr_by_70_per'] == 0:
                return 'FN'
        else:   # row[cfg.CoN_target + '_Recover_home'] == 0:
            if row['Prediction_by_70_per'] == 1:
                return 'FP'
            else:  # row['thr_by_70_per'] == 0:
                return 'TN'

    result_df['TP_TN_FP_FN'] = result_df.apply(TP_TN_FP_FN, axis=1)

    N_bins = 12   # 12 months
    recall_or_precision = 'precision'  # 'recall'   #
    ax = sns.histplot(data=result_df, x="month", hue="TP_TN_FP_FN", multiple="stack", bins=N_bins,
                 hue_order=['FN', 'TP', 'FP', 'TN'], stat='count')   # hue_order is from top to bottom.
    plt.title('Results for 2023 (assume visit top 70% patients)\n(with ' + recall_or_precision + ' for each month)')
    plt.legend(title='results', loc='lower right', labels=['-, pred:-', '-, pred:+', '+,pred:+', '+,pred:-'])

    max_p_all = -1e3
    for N_bin in range(N_bins):
        p_TN = ax.patches[N_bin].get_height()   # TN
        p_FP = ax.patches[N_bin + N_bins *1].get_height()   # FP
        p_TP = ax.patches[N_bin + N_bins *2].get_height()   # TP
        p_FN = ax.patches[N_bin + N_bins *3].get_height()   # FN
        p_all = p_FN + p_TP + p_FP + p_TN
        max_p_all = p_all if max_p_all < p_all else max_p_all
        if recall_or_precision == 'recall':
            rr = (p_TP/(p_TP + p_FN))   # recall
        else:
            rr = (p_TP/(p_TP + p_FP))   # precision
        ax.annotate(str(rr)[:4], (ax.patches[N_bin].get_x() + ax.patches[N_bin].get_width() / 2., p_all),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.tight_layout()
    plt.ylim((0, max_p_all + 40))
    plt.savefig("fig_TP_TN_FP_FN__" + file_str + '_' + str(rand_num) + ".png")
    plt.close()


def generate_cv_strategy(df):
    """ A function that customized cross validation dataset """
    tmp_df = pd.DataFrame(df[cfg.CoN_kfold].unique(), columns=[cfg.CoN_kfold])
    tmp_df[cfg.CoN_kfold_inx] = np.random.randint(0, cfg.n_fold, tmp_df.shape[0])   # the integer: 0, 1, ..., n_fold-1
    df_kfold = df.merge(tmp_df, how='left', on=cfg.CoN_kfold)
    myCViterator = []
    for i in range(0, cfg.n_fold):
        trainIndices = df_kfold[df_kfold[cfg.CoN_kfold_inx] != i].index.values.astype(int)
        testIndices = df_kfold[df_kfold[cfg.CoN_kfold_inx] == i].index.values.astype(int)
        myCViterator.append((trainIndices, testIndices))
    return myCViterator


def SearchBestParameter(df_train, L_process_coef, X_train, Y_train, X_valid, Y_valid, rand_num, SearchType='Grid'):
    """
    Goal: Search the best parameter setting with Grid Search or Bayes Optimization.
    """
    # ..............
    cv_strategy = generate_cv_strategy(df_train)   # generate n-fold based on the PAT_ID
    scoring = make_scorer(roc_auc_score, needs_proba=True, greater_is_better=True)   #     # scoring='auc'

    # ......... https://stackoverflow.com/questions/50686645/grid-search-with-lightgbm-example
    def convert_to_category_type(df):
        # convert into categorical feature for lightgbm
        for CoN in L_process_coef:
            if L_process_coef[CoN][0] == 'cat':
                df.loc[:, CoN] = df.loc[:, CoN].astype('category')

    # ........ don't use category in the set
    fit_params = {'eval_set': [(X_valid, Y_valid.values.ravel())]}   # input the validation set here in the optimization

    # ...... pick onf of the parameter search (Grid Search, Randomize Search, or Bayes optimization Search ........
    if SearchType == 'Grid':
        model_basic = claim_model_obj(cfg.basic_para)
        searchCV = GridSearchCV(estimator=model_basic, param_grid=cfg.model_param, cv=cv_strategy, scoring=scoring)
        searchCV.fit(X_train, Y_train.values.ravel(), **fit_params)

    elif 'Bayes':
        model_basic = claim_model_obj({})
        searchCV = BayesSearchCV(estimator=model_basic
                            , search_spaces=cfg.Bayes_search_space
                            , scoring=scoring   # 'accuracy' ??
                            , cv=cv_strategy   # or cv = 5
                            , n_iter=60  # 60,  # max number of trials
                            , n_points=1   # 3,  # number of hyperparameter sets evaluated at the same time
                            # , n_jobs=-1  # number of jobs  # default is 1
                            # , iid=False  # if not iid it optimizes on the cv score
                            , return_train_score=False
                            # , refit=False
                            # , optimizer_kwargs={'base_estimator': 'GP'}  # optmizer parameters: we use Gaussian Process (GP)
                            , random_state=42
                            , fit_params=fit_params.update(cfg.basic_para)   # evaluation set + basic paramter. This is input during fit
                            # , callback=[early_stopping, time_limit_control]
                            )  # random state for reproducing results.

        searchCV.fit(X_train, Y_train.values.ravel())

    return searchCV


def SHAP_figures_and_find_threshold(model_best_para, X_train, file_str, rand_num, L_process_coef):
    """
    Generate SHAP summery plot.
    """
    shap_values = shap.TreeExplainer(model_best_para).shap_values(X_train)
    shap.summary_plot(shap_values[1], X_train, feature_names=X_train.columns, show=False, max_display=cfg.topN_shap)
    plt.savefig('fig_shap_' + file_str + '_' + str(rand_num) + '.png')
    plt.close()

    # impact_min_abs_shap_value = 0.005
    # N_data = X_train.shape[0]
    # df_shap = pd.DataFrame(
    #     columns=['feature', 'mean_abs_shap_value', 'min_shap_value', 'max_shap_value',
    #              'data_%__neg_shap', 'data_%__pos_shap',
    #              'neg_shap_10%', 'neg_shap_25%', 'neg_shap_median', 'neg_shap_75%', 'neg_shap_90%',
    #              'pos_shap_10%', 'pos_shap_25%', 'pos_shap_median', 'pos_shap_75%', 'pos_shap_90%',
    #              'data_threshold'])
    # for i, feature in enumerate(X_train.columns):
    #     shap_values__values = shap_values[1][:, i]
    #     shap_values__data = np.array(X_train.iloc[:, i])
    #
    #     df_shap.loc[i, 'feature'] = feature
    #     df_shap.loc[i, 'mean_abs_shap_value'] = np.mean(np.abs(shap_values__values))
    #     df_shap.loc[i, 'max_shap_value'] = np.max(shap_values__values)
    #     df_shap.loc[i, 'min_shap_value'] = np.min(shap_values__values)
    #
    #     # ...... extract the index of data with real data (i.e., non-missing values) ......
    #     if L_process_coef[feature][1] == '':  # if there is no missing value
    #         inx_non_missing_data = np.array([True] * N_data)
    #     else:
    #         if L_process_coef[feature][0] == 'num':
    #             inx_non_missing_data = shap_values__data != L_process_coef[feature][1]   # missing value
    #         elif L_process_coef[feature][0] == 'cat':
    #             inx_non_missing_data = shap_values__data != list(L_process_coef[feature][2]).index(L_process_coef[feature][1])
    #     # ..................
    #     inx_data_pos_shap = (shap_values__values > impact_min_abs_shap_value) & inx_non_missing_data
    #     inx_data_neg_shap = (shap_values__values < -1 * impact_min_abs_shap_value) & inx_non_missing_data
    #
    #     data_percent__pos_shap = inx_data_pos_shap.sum() / N_data * 100
    #     df_shap.loc[i, 'data_%__pos_shap'] = data_percent__pos_shap
    #     data_percent__neg_shap = inx_data_neg_shap.sum() / N_data * 100
    #     df_shap.loc[i, 'data_%__neg_shap'] = data_percent__neg_shap
    #
    #     if data_percent__pos_shap > 0.1:   # only calculate statistics when there is data
    #         tmp = shap_values__data[inx_data_pos_shap]   # the data for the positive shap (above certian threshold)
    #         df_shap.loc[i, 'pos_shap_10%'] = np.percentile(tmp, 10)
    #         df_shap.loc[i, 'pos_shap_25%'] = np.percentile(tmp, 25)
    #         df_shap.loc[i, 'pos_shap_median'] = np.median(tmp)
    #         df_shap.loc[i, 'pos_shap_75%'] = np.percentile(tmp, 75)
    #         df_shap.loc[i, 'pos_shap_90%'] = np.percentile(tmp, 90)
    #
    #     if data_percent__neg_shap > 0.1:  # only calculate statistics when there is data
    #         tmp = shap_values__data[inx_data_neg_shap]   # the data for the positive shap (above certian threshold)
    #         df_shap.loc[i, 'neg_shap_10%'] = np.percentile(tmp, 10)
    #         df_shap.loc[i, 'neg_shap_25%'] = np.percentile(tmp, 25)
    #         df_shap.loc[i, 'neg_shap_median'] = np.median(tmp)
    #         df_shap.loc[i, 'neg_shap_75%'] = np.percentile(tmp, 75)
    #         df_shap.loc[i, 'neg_shap_90%'] = np.percentile(tmp, 90)
    #
    #     # only calculate threshold when there are sufficient data
    #     if data_percent__pos_shap > 2 and data_percent__neg_shap > 2:  # make sure both label exist > 3% of data
    #         df_shap.loc[i, 'data_threshold'] = find_threshold_through_logistic_regression(   # output the data threshold
    #             shap_values__data[(inx_data_pos_shap | inx_data_neg_shap)].reshape(-1, 1),   # by input "data" &
    #             ((shap_values__values[(inx_data_pos_shap | inx_data_neg_shap)]) > 0) + 0)    # "binary label" (positive or negative influence)
    #
    # # should remove the close to zero values, and split the values into ">0" & "<0", and find the threshold.
    # df_shap = df_shap.sort_values(by='mean_abs_shap_value', ascending=False)
    # df_shap.to_csv('df_shap' + file_str + '_' + str(rand_num) + '.csv')

    return shap_values


def model_train(df_train, df_valid, df_test, rand_num, file_str, df_best_para):
    """
    Goal: train a model, evaluate it, and save results.

    Args:
        df_train: Dataframe of train data.
        df_valid: Dataframe of validation data.
        df_test: Dataframe of test data.
        rand_num: The seed for random_state.
        file_str: The string, which denotes model properties for this round, and used in the file name.
        df_best_para: A dataframe that records the best parameters for the model for each round with different rand_num.

    Returns:
        The model performance matrix:
        mae: Mean Absolute Error.
        AUC_RecallPrecision: AUC of Recall Precision plot.
        roc_auc: AUC of roc curve.
    """

    X_train, X_train_with_ID, Y_train, Y_train_with_ID, L_process_coef = data_pre_process(df_train, 'train', [])
    X_valid, X_valid_with_ID, Y_valid, Y_valid_with_ID, _ = data_pre_process(df_valid, 'test', L_process_coef)

    X_train_with_ID[:1000].to_csv('Model_output__df_X_train_with_ID' + file_str + '.csv')
    Y_train_with_ID[:1000].to_csv('Model_output__df_Y_train_with_ID' + file_str + '.csv')

    # =============== could customize the grid search and cross validation data split ===============
    # ---- add categorical feature into lightgbm parameter set ------
    searchCV = SearchBestParameter(df_train, L_process_coef, X_train, Y_train, X_valid, Y_valid
                                   , rand_num, SearchType='Bayes')
    best_para = cfg.basic_para | searchCV.best_params_
    model_best_para_ = claim_model_obj(best_para)
    if (cfg.model_type == 'xgb') or (cfg.model_type == 'lightgbm'):
        model_best_para_.fit(X_train, Y_train, eval_set=[(X_valid, Y_valid)])
    elif cfg.model_type == 'rf':
        model_best_para_.fit(X_train, Y_train)

    best_iter = model_best_para_.best_iteration_
    best_para['n_estimators'] = best_iter
    for para in best_para:
        df_best_para.loc[rand_num, para] = best_para[para]
    df_best_para.to_csv('Model_output__best_para' + file_str + '.csv')

    del best_para['early_stopping_rounds']
    model_best_para = claim_model_obj(best_para)
    model_best_para.fit(X_train, Y_train)

    # ================ before having shap results, save the model first ============
    shap_values = SHAP_figures_and_find_threshold(model_best_para, X_train, file_str, rand_num, L_process_coef)
    if not cfg.Q_evaluate_model_Q:
        model_data = [model_best_para, L_process_coef]
    else:
        model_data = [model_best_para, L_process_coef, shap_values]  # , X_test]
    pickle.dump(model_data, open("model" + file_str + '_' + str(rand_num) + ".bin", "wb"))

    # ...... feature importance for the lightgbm ......
    lgb.plot_importance(model_best_para, max_num_features=30)  # bst
    plt.title("Feature Importance");
    plt.tight_layout()
    plt.savefig('fig_fea_importance_' + file_str + '_' + str(rand_num) + '.png')
    plt.close()

    if not cfg.Q_evaluate_model_Q:
        if cfg.model_target_type == 'regression':
            Y_pred = model_best_para.predict(X_train)  # the prediction of being 1
        else:  # classification, predict_proba for binary_classification first.
            Y_pred = model_best_para.predict_proba(X_train)[:, 1]

        mae = mean_absolute_error(Y_train.astype('float').values, Y_pred)
        AUC_RecallPrecision = average_precision_score(np.array(Y_train[cfg.CoN_target]), Y_pred)
        roc_auc = roc_auc_score(np.array(Y_train[cfg.CoN_target]), Y_pred)
        print('Performance in training data: mae=' + str(mae))
        print('Performance in training data: roc_auc=' + str(roc_auc))
        print('Performance in training data: AUC_RecallPrecision=' + str(AUC_RecallPrecision))

    # @@@@@@@@@@@@@@@@@@@@@@@@ with model, evaluate performance (accuracy, AUC plots, etc.) @@@@@@@@@@@@@@@@@@
    if cfg.Q_evaluate_model_Q:
        X_test, X_test_with_ID, Y_test, Y_test_with_ID, _ = data_pre_process(df_test, 'test', L_process_coef)

        if cfg.model_target_type == 'regression':
            Y_pred = model_best_para.predict(X_test)  # the prediction of being 1
        else:  # classification, predict_proba for binary_classification first.
            Y_pred = model_best_para.predict_proba(X_test)[:, 1]

        mae = mean_absolute_error(Y_test.astype('float').values, Y_pred)
        AUC_RecallPrecision = average_precision_score(np.array(Y_test[cfg.CoN_target]), Y_pred)
        roc_auc = roc_auc_score(np.array(Y_test[cfg.CoN_target]), Y_pred)
        print('mae=' + str(mae))
        print('roc_auc=' + str(roc_auc))
        print('AUC_RecallPrecision=' + str(AUC_RecallPrecision))

        # ================ output prediction for all test data ================
        Y_test_with_ID.insert(loc=0, column='Prediction', value=Y_pred)
        X_test_with_ID.insert(loc=0, column='Prediction', value=Y_pred)

        if rand_num == cfg.List_rand_num[0]:   # just output an sample
            Y_test_with_ID.to_csv('Model_output__Y_with_prediction' + file_str + '.csv')
            X_test_with_ID[:1000].to_csv('Model_output__sampled_X_with_prediction' + file_str + '.csv')

            tmp = Y_test_with_ID[['Prediction', 'PAT_ID', 'VISIT_NO']].copy()
            # tmp.groupby(by=['PAT_ID', 'VISIT_NO']).head(2)
            tmp2 = tmp.groupby(by=['PAT_ID'], as_index=False).head(1)  # demo with the first prediction for each patient
            tmp2 = tmp2.sort_values(by='Prediction', ascending=False).reset_index(drop=True)
            tmp2.to_csv('Model_output__prediction_table.csv')

            tmp3 = tmp2.sample(frac=1.0).sample(n=20, random_state=rand_num)
            tmp3 = tmp3.sort_values(by='Prediction', ascending=False).reset_index(drop=True)
            tmp3.to_csv('Model_output__prediction_table_20samples.csv')

        # @@@@@@@@@@@@@@@ (simple version) performance plot @@@@@@@@@@@@@@@@
        # ....... performance plot, roc_AUC & precision recall .................
        mae = mean_absolute_error(Y_test.astype('float').values, Y_pred)
        AUC_RecallPrecision = average_precision_score(np.array(Y_test[cfg.CoN_target]), Y_pred)
        roc_auc = roc_auc_score(Y_test, Y_pred)

        precision, recall, thresholds = precision_recall_curve(np.array(Y_test[cfg.CoN_target]), Y_pred)
        plt.figure(figsize=(7, 5))
        plt.plot(recall, precision, 'r')
        plt.title('recall_precision: mae=' + str(mae)[:4] + ', roc_auc=' + str(roc_auc)[:4] +\
                  ', Avg. precision=' + str(AUC_RecallPrecision)[:4])
        plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.ylim([-0.02, 1.02]); plt.xlim([-0.02, 1.02]); plt.grid()
        plt.savefig('_fig_PRcurve' + file_str + '_' + str(rand_num) + '.png')
        plt.close()

        fpr, tpr, thresholds = roc_curve(np.array(Y_test[cfg.CoN_target]), Y_pred)
        roc_auc = roc_auc_score(np.array(Y_test[cfg.CoN_target]), Y_pred)
        plt.plot(fpr, tpr, 'r')
        plt.title('ROC: AUC = ' + str(roc_auc)[:4] + ', mae=' + str(mae)[:4])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.ylim([-0.02, 1.02]); plt.xlim([-0.02, 1.02]); plt.grid()
        plt.savefig('_fig_roc_' + file_str + '_' + str(rand_num) + '.png')
        plt.close()

        print('CoN_target = ' + cfg.CoN_target)
        print(file_str)
        print("")
        print('mae=' + str(mae))
        print('AUC_RecallPrecision=' + str(AUC_RecallPrecision))
        print('roc_auc=' + str(roc_auc))
        print(cfg.basic_para)
        print(searchCV.best_params_)
        print('Yuli')

        # ======= (complicated version) performance plot =====
        # (scatter plot for regression & recall_precision plot for binary classifier)
        if cfg.model_target_type == 'regression':
            plt.scatter(np.array(Y_test[cfg.CoN_target]), Y_pred, color='r', label='pred')  # , label='Data Points')
            tmp_m = Y_pred.min()
            tmp_M = Y_pred.max()
            plt.plot([tmp_m, tmp_M], [tmp_m, tmp_M], linestyle='--', color='green', label='ref')
            # plot_h = sns.jointplot(data=Y_test_with_ID, x=cfg.CoN_target + '_Recover_home', y='Prediction')
            # fig = plot_h.fig
            # plt.legend(loc='upper right')
            # plt.tight_layout()
            RMSE = math.sqrt(np.mean((np.array(Y_test[cfg.CoN_target]) - Y_pred)**2))    # L2 Error
            corr_coef = np.corrcoef(np.array(Y_test[cfg.CoN_target]), Y_pred)[0, 1]    # df['x'].corr(df['y'])
            plt.title('Performance (RMSE=' + str(RMSE)[:4] + ', corr=' + str(corr_coef)[:4] + ')\n' + \
                       ' (--: baseline. data_rand_num:' + str(rand_num) + ')')
            plt.xlabel('Truth')
            plt.ylabel('Prediction')
            plt.savefig('fig_reg_perf_' + file_str + '_' + str(rand_num) + '.png')
            plt.close()    # close specific sns figure: plt.close(sns_plot.fig)

        elif cfg.model_target_type == 'binary_class':   # classification, predict_proba for binary_classification first.
            accuracy = accuracy_score((Y_test>0.5).values, Y_pred>0.5) * 100  # need to be both are integer (label)
            mae = mean_absolute_error(Y_test.astype('float').values, Y_pred)
            print("ACCURACY: " + str(accuracy) + " %")
            print("mae: ", str(mae))
            precision, recall, thresholds = precision_recall_curve(Y_test, Y_pred)
            AUC_RecallPrecision = average_precision_score(Y_test, Y_pred)
            roc_auc = roc_auc_score(Y_test, Y_pred)

            list_predict_random__ = (np.random.sample(Y_pred.shape[0] * 100)).tolist()  # random.sample(range(10, 30), 5)
            precision_r, recall_r, thresholds_r = \
                precision_recall_curve(Y_test[cfg.CoN_target].to_list()*100, list_predict_random__)

            # ----- find the recall & precision when we could cover 70% of patient daily, in general -----
            # therapy could cover ~ 70 % of patient in general.
            Y_pred_sort = sorted(Y_pred, reverse=True)
            Y_pred_thr = Y_pred_sort[int(len(Y_pred_sort) * cfg.Rehab_staff_cover)]   # possible threshold based on the top 70% patients, above this  threshold is picked.

            close_inx = -1
            close_tt = -1
            close_diff = 1e3
            for inx in range(len(thresholds)):
                tt = thresholds[inx]
                if close_diff > abs(tt - Y_pred_thr):
                    close_inx = inx
                    close_tt = tt
                    close_diff = abs(tt - Y_pred_thr)

            print(close_tt)
            print(close_inx)
            print(close_diff)

            f1_score_close = 2*recall[close_inx]*precision[close_inx]/(recall[close_inx] + precision[close_inx])
            # -------------------- model performance base on all records for each available recorded date -----
            plt.figure(figsize=(7, 5))
            plt.plot(recall, precision, 'r', recall_r, precision_r, 'g--', recall[close_inx], precision[close_inx], 'ro')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Performance (70% staff: recall=' + str(recall[close_inx])[:4] + \
                      ', precision=' + str(precision[close_inx])[:4] + \
                      ', f1 score=' + str(f1_score_close)[:4] + ')\n<Overall> roc_AUC:' + str(roc_auc)[:5] + \
                      ' rp_AUC:' + str(AUC_RecallPrecision)[:5] + ', mae: ' + str(mae)[:4]+ \
                       ' (--: baseline,' + str(rand_num) + ')')
            plt.ylim([-0.02, 1.02])
            plt.xlim([-0.02, 1.02])
            plt.grid()
            plt.savefig('fig_PRcurve' + file_str + '_' + str(rand_num) + '.png')
            plt.close()

            # # ------- visualize the TP, FP, TN, FN -----------
            # visualize_TP_FP_TN_FN_per_month(X_test_with_ID, rand_num, file_str)
            # tmp = X_test_with_ID.loc[X_test_with_ID['DIFF_DATE_PRIOR1_FST'] <= 3, :]   # if the previuos record date - first record date >= 3 days. So this make sure at least have first record (admission day could even more earlier)
            # visualize_TP_FP_TN_FN_per_month(tmp, rand_num, file_str+'_EarlyDay')

    return mae, AUC_RecallPrecision, roc_auc
