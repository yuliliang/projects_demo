import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import AcuteRehabRecoverRate_Yuli__config_ as cfg
from AcuteRehabRecoverRate_Yuli__data_query_ import data_pre_process

from sklearn.metrics import (accuracy_score, precision_recall_curve, mean_absolute_error, average_precision_score,
                             roc_curve, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay)

def eval_test_data(model_best_para, df_test, L_process_coef, rand_num, file_str):
    """
    Goal: generate the performance evaluation of a model.

    Args:
        model_best_para: The model trained with the best parameters.
        df_test: Dataframe of test data.
        L_process_coef: A list of coefficients, which records how to pre-process the data.
        rand_num: The seed for random_state.
        file_str: The string, which denotes model properties for this round, and used in the file name.

    Returns:
        The model performance matrix:
        mae: Mean Absolute Error.
        AUC_RecallPrecision: AUC of Recall Precision plot.
        roc_auc: AUC of roc curve.
    """

    # X_train, X_train_with_ID, Y_train, Y_train_with_ID, L_process_coef = data_pre_process(df_train, 'train', [])
    # X_valid, X_valid_with_ID, Y_valid, Y_valid_with_ID, _ = data_pre_process(df_valid, 'test', L_process_coef)
    #
    # X_train_with_ID[:1000].to_csv(cfg.DIR_output + 'Model_output__df_X_train_with_ID' + file_str + '.csv')
    # Y_train_with_ID[:1000].to_csv(cfg.DIR_output + 'Model_output__df_Y_train_with_ID' + file_str + '.csv')
    #
    # # =============== could customize the grid search and cross validation data split ===============
    # # ---- add categorical feature into lightgbm parameter set ------
    # searchCV = SearchBestParameter(df_train, L_process_coef, X_train, Y_train, X_valid, Y_valid
    #                                , rand_num, SearchType='Bayes')
    # best_para = cfg.basic_para | searchCV.best_params_
    # model_best_para_ = claim_model_obj(best_para)
    # if (cfg.model_type == 'xgb') or (cfg.model_type == 'lightgbm'):
    #     model_best_para_.fit(X_train, Y_train, eval_set=[(X_valid, Y_valid)])
    # elif cfg.model_type == 'rf':
    #     model_best_para_.fit(X_train, Y_train)
    #
    # best_iter = model_best_para_.best_iteration_
    # best_para['n_estimators'] = best_iter
    # for para in best_para:
    #     df_best_para.loc[rand_num, para] = best_para[para]
    # df_best_para.to_csv(cfg.DIR_output + 'Model_output__best_para' + file_str + '.csv')
    #
    # del best_para['early_stopping_rounds']
    # model_best_para = claim_model_obj(best_para)
    # model_best_para.fit(X_train, Y_train)
    #
    # # ================ before having shap results, save the model first ============
    # shap_values = SHAP_figures(model_best_para, X_train, file_str, rand_num, L_process_coef)
    # if not cfg.Q_evaluate_model_Q:
    #     model_data = [model_best_para, L_process_coef]
    # else:
    #     model_data = [model_best_para, L_process_coef, shap_values]  # , X_test]
    # pickle.dump(model_data, open("model" + file_str + '_' + str(rand_num) + ".bin", "wb"))
    #
    # # ...... feature importance for the lightgbm ......
    # lgb.plot_importance(model_best_para, max_num_features=30)  # bst
    # plt.title("Feature Importance");
    # plt.tight_layout()
    # plt.savefig(cfg.DIR_output + 'fig_fea_importance_' + file_str + '_' + str(rand_num) + '.png')
    # plt.close()
    #
    # if not cfg.Q_evaluate_model_Q:
    #     if cfg.model_target_type == 'regression':
    #         Y_pred = model_best_para.predict(X_train)  # the prediction of being 1
    #     else:  # classification, predict_proba for binary_classification first.
    #         Y_pred = model_best_para.predict_proba(X_train)[:, 1]
    #
    #     mae = mean_absolute_error(Y_train.astype('float').values, Y_pred)
    #     AUC_RecallPrecision = average_precision_score(np.array(Y_train[cfg.CoN_target]), Y_pred)
    #     roc_auc = roc_auc_score(np.array(Y_train[cfg.CoN_target]), Y_pred)
    #     print('Performance in training data: mae=' + str(mae))
    #     print('Performance in training data: roc_auc=' + str(roc_auc))
    #     print('Performance in training data: AUC_RecallPrecision=' + str(AUC_RecallPrecision))

    # @@@@@@@@@@@@@@@@@@@@@@@@ with model, evaluate performance (accuracy, AUC plots, etc.) @@@@@@@@@@@@@@@@@@
    X_test, X_test_with_ID, Y_test, Y_test_with_ID, _ = data_pre_process(df_test, 'test', L_process_coef)

    # ----- check if column of data is consistent with the model features (& align the order) -------
    fea_cols = model_best_para.booster_.feature_name()
    for inx, val in enumerate(list(X_test.columns)):
        if (list(X_test.columns))[inx] != fea_cols[inx]:
            print('inconsistent feature order: ' + str(inx) + ', ' + str(val))
    X_test = X_test.reindex(columns=fea_cols)

    X_test_with_ID[:1000].to_csv(cfg.DIR_output + 'Model_output__df_X_test_with_ID' + file_str + '.csv')
    Y_test_with_ID[:1000].to_csv(cfg.DIR_output + 'Model_output__df_Y_test_with_ID' + file_str + '.csv')

    if cfg.model_target_type == 'regression':
        Y_pred = model_best_para.predict(X_test)  # the prediction of being 1
        mae = mean_absolute_error(Y_test.astype('float').values, Y_pred)
        print('mae=' + str(mae))

        plt.scatter(np.array(Y_test[cfg.CoN_target]), Y_pred, color='r', label='pred')
        tmp_m = Y_pred.min()
        tmp_M = Y_pred.max()
        plt.plot([tmp_m, tmp_M], [tmp_m, tmp_M], linestyle='--', color='green', label='ref')
        # plot_h = sns.jointplot(data=Y_test_with_ID, x=cfg.CoN_target, y='Prediction')
        # fig = plot_h.fig; plt.legend(loc='upper right'); plt.tight_layout()
        RMSE = math.sqrt(np.mean((np.array(Y_test[cfg.CoN_target]) - Y_pred) ** 2))  # L2 Error
        corr_coef = np.corrcoef(np.array(Y_test[cfg.CoN_target]), Y_pred)[0, 1]  # df['x'].corr(df['y'])
        plt.title('Performance (RMSE=' + str(RMSE)[:4] + ', corr=' + str(corr_coef)[:4] + ')\n' + \
                  ' (--: baseline. data_rand_num:' + str(rand_num) + ')')
        plt.xlabel('Truth')
        plt.ylabel('Prediction')
        plt.savefig(cfg.DIR_output + 'fig_reg_perf_' + file_str + '_' + str(rand_num) + '.png')
        plt.close()  # close specific sns figure: plt.close(sns_plot.fig)
        return mae

    elif cfg.model_target_type == 'binary_class':  # classification, predict_proba for binary_classification first.
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
            Y_test_with_ID.to_csv(cfg.DIR_output + 'Model_output__Y_with_prediction' + file_str + '.csv')
            X_test_with_ID[:1000].to_csv(cfg.DIR_output + 'Model_output__sampled_X_with_prediction' + file_str + '.csv')

            # tmp = Y_test_with_ID[['Prediction', 'PAT_ID', 'VISIT_NO']].copy()
            # # tmp.groupby(by=['PAT_ID', 'VISIT_NO']).head(2)
            # tmp2 = tmp.groupby(by=['PAT_ID'], as_index=False).head(1)  # demo with the first prediction for each patient
            # tmp2 = tmp2.sort_values(by='Prediction', ascending=False).reset_index(drop=True)
            # tmp2.to_csv(cfg.DIR_output + 'Model_output__prediction_table.csv')
            #
            # tmp3 = tmp2.sample(frac=1.0).sample(n=20, random_state=rand_num)
            # tmp3 = tmp3.sort_values(by='Prediction', ascending=False).reset_index(drop=True)
            # tmp3.to_csv(cfg.DIR_output + 'Model_output__prediction_table_20samples.csv')

        # @@@@@@@@@@@@@@@ Performance plot (Basic version) @@@@@@@@@@@@@@@@
        # ....... performance plot, roc_AUC & precision recall .................
        mae = mean_absolute_error(Y_test.astype('float').values, Y_pred)
        AUC_RecallPrecision = average_precision_score(np.array(Y_test[cfg.CoN_target]), Y_pred)
        roc_auc = roc_auc_score(Y_test, Y_pred)

        precision, recall, thresholds = precision_recall_curve(np.array(Y_test[cfg.CoN_target]), Y_pred)
        plt.figure(figsize=(7, 5))
        plt.plot(recall, precision, 'r')
        plt.title('Recall_Precision: AUC (avg. precision)=' + str(AUC_RecallPrecision)[:4] + \
                  ', mae=' + str(mae)[:4] + ', roc_auc=' + str(roc_auc)[:4])
        plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.ylim([-0.02, 1.02]); plt.xlim([-0.02, 1.02]); plt.grid()
        plt.savefig(cfg.DIR_output + '_fig_PRcurve' + file_str + '_' + str(rand_num) + '.png')
        plt.close()

        fpr, tpr, thresholds = roc_curve(np.array(Y_test[cfg.CoN_target]), Y_pred)
        roc_auc = roc_auc_score(np.array(Y_test[cfg.CoN_target]), Y_pred)
        plt.plot(fpr, tpr, 'r')
        plt.title('ROC: AUC = ' + str(roc_auc)[:4] + ', mae=' + str(mae)[:4])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.ylim([-0.02, 1.02]); plt.xlim([-0.02, 1.02]); plt.grid()
        plt.savefig(cfg.DIR_output + '_fig_roc_' + file_str + '_' + str(rand_num) + '.png')
        plt.close()

        print('CoN_target = ' + cfg.CoN_target)
        print(file_str)
        print("")
        print('mae=' + str(mae))
        print('AUC_RecallPrecision=' + str(AUC_RecallPrecision))
        print('roc_auc=' + str(roc_auc))
        print(cfg.basic_para)
        # print(searchCV.best_params_)

        # ======= performance plot (Customized version)  =====
        # (scatter plot for regression & recall_precision plot for binary classifier)
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
        plt.savefig(cfg.DIR_output + 'fig_PRcurve' + file_str + '_' + str(rand_num) + '.png')
        plt.close()

        # # ------- visualize the TP, FP, TN, FN -----------
        # visualize_TP_FP_TN_FN_per_month(X_test_with_ID, rand_num, file_str)
        # tmp = X_test_with_ID.loc[X_test_with_ID['DIFF_DATE_PRIOR1_FST'] <= 3, :]   # if the previuos record date - first record date >= 3 days. So this make sure at least have first record (admission day could even more earlier)
        # visualize_TP_FP_TN_FN_per_month(tmp, rand_num, file_str+'_EarlyDay')
        return mae, AUC_RecallPrecision, roc_auc

    elif cfg.model_target_type == 'multi_class':
        y_pred = model_best_para.predict(X_test)
        accuracy = accuracy_score(Y_test, y_pred)
        report = classification_report(Y_test, y_pred, output_dict=True)
        cm = confusion_matrix(Y_test, y_pred)

        print(accuracy)
        pd.DataFrame(report).transpose()\
              .to_excel(cfg.DIR_output + "classification_report_" + file_str + '_' + str(rand_num) + ".xlsx")

        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=model_best_para.classes_)
        disp.plot()  # plt.show()
        plt.savefig(cfg.DIR_output + 'Confusion_matrix_' + file_str + '_' + str(rand_num) + '.png')
        plt.close()
        return accuracy

    else:
        print("The value of model_target_type is not valid: " + str(cfg.model_target_type))

    return 1
