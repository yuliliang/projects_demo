import os.path
import pickle, shap
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
import AcuteRehabRecoverRate_Yuli__config_ as cfg
from AcuteRehabRecoverRate_Yuli__data_query_ import data_pre_process
from sklearn.metrics import (accuracy_score, mean_absolute_error, average_precision_score, roc_auc_score, make_scorer,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)


def load_previous_model(path_previous_model):
    model_data = pickle.load(open(path_previous_model, "rb"))    # .load_model()
    model_best_para = model_data[0]
    L_process_coef = model_data[1]
    if cfg.Q_evaluate_model_Q:
        shap_values = model_data[2]
    return model_best_para, L_process_coef


def claim_model_obj(para):
    """
    Goal: Claim the correct model object.
    """
    # if cfg.model_type == 'xgb':
    #   return xgb.XGBClassifier(**para)  # Train data using XGBRegressor with the best parameter
    # elif cfg.model_type == 'lightgbm':
    #   if cfg.model_target_type == 'regression':
    #     return lgb.LGBMRegressor(**para)
    #   else:
    return lgb.LGBMClassifier(**para)
    # elif cfg.model_type == 'rf':
    #   return RandomForestClassifier(**para)


def generate_cv_strategy(df, CoN_kfold=''):
    """ A function that customized cross validation dataset """
    tmp_df = pd.DataFrame(df[CoN_kfold].unique(), columns=[CoN_kfold])
    tmp_df[cfg.CoN_kfold_inx] = np.random.randint(0, cfg.n_fold, tmp_df.shape[0])  # kfold_inx: 0, 1, ..., n_fold-1
    df_kfold = df.merge(tmp_df, how='left', on=CoN_kfold)

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
    if cfg.CoN_kfold == '':   # The general cross validation split
      cv_strategy = cfg.n_fold
    else:  # split the data into k-fold based on values in 1 certain column.
      cv_strategy = generate_cv_strategy(df_train, CoN_kfold=cfg.CoN_kfold)  # generate n-fold based on the PAT_ID

    # scoring = 'accuracy'   # With this & if the scoring didn't refresh below, it causes issue.
    if cfg.model_target_type == 'binary_class':
        scoring = make_scorer(roc_auc_score, needs_proba=True, greater_is_better=True)  # # scoring='auc'
    elif cfg.model_target_type == 'multi_class':
        scoring = make_scorer(roc_auc_score, needs_proba=True, greater_is_better=True,
                              multi_class='ovr', average='micro')
        # multi_class: 'raise, 'ovr', 'ovo'
        # ovo (one-vs-one): here could use ovo, since all possible pairs were using, and here only have 3 multi-class.
        # average = 'macro', 'micro', 'weighted', 'samples', it seems only work for ovr, not ovo

        # scoring = make_scorer(roc_auc_score, needs_proba=True, greater_is_better=True, multi_class='ovo')  # # scoring='auc'   TODO: ???? fix

    # ......... https://stackoverflow.com/questions/50686645/grid-search-with-lightgbm-example
    def convert_to_category_type(df):
        # convert into categorical feature for lightgbm
        for CoN in L_process_coef:
            if L_process_coef[CoN][0] == 'cat':
                df.loc[:, CoN] = df.loc[:, CoN].astype('category')

    # ........ don't use category in the set .....
    fit_params = {'eval_set': [(X_valid, Y_valid.values.ravel())]}  # input the validation set here in the optimization

    # ...... pick onf of the parameter search (Grid Search, Randomize Search, or Bayes optimization Search ........
    if SearchType == 'Grid':
        model_basic = claim_model_obj(cfg.basic_para)
        searchCV = GridSearchCV(estimator=model_basic, param_grid=cfg.model_param, cv=cv_strategy, scoring=scoring)
        searchCV.fit(X_train, Y_train.values.ravel(), **fit_params)

    elif 'Bayes':
        model_basic = claim_model_obj({})
        searchCV = BayesSearchCV(estimator=model_basic
                                 , search_spaces=cfg.Bayes_search_space
                                 , scoring=scoring  # 'accuracy' ??
                                 , cv=cv_strategy  # or cv = 5
                                 , n_iter=60  # 60,  # max number of trials
                                 , n_points=1  # 3,  # number of hyperparameter sets evaluated at the same time
                                 # , n_jobs=-1  # number of jobs  # default is 1
                                 # , iid=False  # if not iid it optimizes on the cv score
                                 , return_train_score=False
                                 # , refit=False
                                 # , optimizer_kwargs={'base_estimator': 'GP'}  # optmizer parameters: we use Gaussian Process (GP)
                                 , random_state=42
                                 , fit_params=fit_params.update(cfg.basic_para)
                                 # evaluation set + basic parameter. This is input during fit
                                 # , callback=[early_stopping, time_limit_control]
                                 )  # random state for reproducing results.

        searchCV.fit(X_train, Y_train.values.ravel())

    return searchCV


def SHAP_figures(model_best_para, X_train, file_str, rand_num, L_process_coef):
    shap_values = shap.TreeExplainer(model_best_para).shap_values(X_train)
    if cfg.model_target_type == 'binary_class':
        shap.summary_plot(shap_values[1], X_train, feature_names=X_train.columns, show=False, max_display=cfg.topN_shap)
    elif cfg.model_target_type == 'multi_class':
        shap.summary_plot(shap_values, X_train, feature_names=X_train.columns, show=False, max_display=cfg.topN_shap,
                          class_names=model_best_para.classes_)
    plt.savefig(cfg.DIR_output + 'fig_shap_' + file_str + '_' + str(rand_num) + '.png')
    plt.close()
    return shap_values


def model_train(df_train, df_valid, rand_num, file_str, df_best_para):
    """
    Goal: train a model with the best parameters & save it.

    Args:
        df_train: Dataframe of train data.
        df_valid: Dataframe of validation data.
        rand_num: The seed for random_state.
        file_str: The string, which denotes model properties for this round, and used in the file name.
        df_best_para: A dataframe that records the best parameters for the model for each round with different rand_num.

    Returns:
        model_best_para: The model trained with the best parameters.
        L_process_coef: A list of coefficients, which records how to pre-process the data.
        model_path_FName: Path and file name of the saved model. The full path is model_path_FName + '.bin"
    """
    X_train, X_train_with_ID, Y_train, Y_train_with_ID, L_process_coef = \
        (data_pre_process(df_train, 'train', []))
    X_valid, X_valid_with_ID, Y_valid, Y_valid_with_ID, _ = \
        (data_pre_process(df_valid, 'test', L_process_coef))

    X_train_with_ID[:1000].to_csv(cfg.DIR_output +'Model_output__df_X_train_with_ID' + file_str + '.csv')
    Y_train_with_ID[:1000].to_csv(cfg.DIR_output +'Model_output__df_Y_train_with_ID' + file_str + '.csv')

    # =============== could customize the grid search and cross validation data split ===============
    # ---- add categorical feature into lightgbm parameter set ------
    searchCV = SearchBestParameter(df_train, L_process_coef, X_train, Y_train, X_valid, Y_valid
                                   , rand_num, SearchType='Bayes')
    best_para = cfg.basic_para | searchCV.best_params_
    # model_best_para = searchCV.best_estimator_

    # ..... using all the df_train and df_valid to have a final model, with the best parameter set .......
    model_best_para_ = claim_model_obj(best_para)
    if (cfg.model_type == 'xgb') or (cfg.model_type == 'lightgbm'):
        model_best_para_.fit(X_train, Y_train, eval_set=[(X_valid, Y_valid)])
    elif cfg.model_type == 'rf':
        model_best_para_.fit(X_train, Y_train)

    best_iter = model_best_para_.best_iteration_
    best_para['n_estimators'] = best_iter
    for para in best_para:
        df_best_para.loc[rand_num, para] = best_para[para]
    df_best_para.to_csv(cfg.DIR_output +'Model_output__best_para' + file_str + '.csv')

    del best_para['early_stopping_rounds']
    model_best_para = claim_model_obj(best_para)
    model_best_para.fit(X_train, Y_train)

    # ================ before having shap results, save the model first ============
    shap_values = SHAP_figures(model_best_para, X_train, file_str, rand_num, L_process_coef)
    if not cfg.Q_evaluate_model_Q:
        model_data = [model_best_para, L_process_coef]
    else:
        model_data = [model_best_para, L_process_coef, shap_values]  # , X_test]
    model_file_name = "model" + file_str + '_' + str(rand_num)
    if os.path.exists(cfg.DIR_output + model_file_name + ".bin"):  # if the model file exist, save it as previous version
        if os.path.exists(cfg.DIR_output + model_file_name + "_pre.bin"):
            os.remove(cfg.DIR_output + model_file_name + "_pre.bin")
        os.rename(cfg.DIR_output + model_file_name + ".bin", cfg.DIR_output + model_file_name + "_pre.bin")
    pickle.dump(model_data, open(cfg.DIR_output + model_file_name + ".bin", "wb"))
    model_path_FName = cfg.DIR_output + model_file_name

    # ...... feature importance for the lightgbm ......
    lgb.plot_importance(model_best_para, max_num_features=30)  # bst
    plt.title("Feature Importance");
    plt.tight_layout()
    plt.savefig(cfg.DIR_output + 'fig_fea_importance_' + file_str + '_' + str(rand_num) + '.png')
    plt.close()

    if cfg.Q_evaluate_train_perf_Q:
        file_str += '_TrainPerf'
        Y_train_val = Y_train.astype('float').values

        if cfg.model_target_type == 'regression':
            Y_pred = model_best_para.predict(X_train)  # the prediction of being 1
            mae = mean_absolute_error(Y_train_val, Y_pred)
            print('Performance in training data: mae=' + str(mae))

        elif cfg.model_target_type == 'binary_class':  # classification, predict_proba for binary_classification first.
            Y_pred = model_best_para.predict_proba(X_train)[:, 1]
            mae = mean_absolute_error(Y_train_val, Y_pred)   # Y_train_val
            AUC_RecallPrecision = average_precision_score(np.array(Y_train[cfg.CoN_target]), Y_pred)
            roc_auc = roc_auc_score(np.array(Y_train[cfg.CoN_target]), Y_pred)
            print('Performance in training data: mae=' + str(mae))
            print('Performance in training data: roc_auc=' + str(roc_auc))
            print('Performance in training data: AUC_RecallPrecision=' + str(AUC_RecallPrecision))

        elif cfg.model_target_type == 'multi_class':
            y_pred = model_best_para.predict(X_train)
            accuracy = accuracy_score(Y_train, y_pred)
            report = classification_report(Y_train, y_pred, output_dict=True)
            # conf_matrix = confusion_matrix(Y_train, y_pred)
            cm = confusion_matrix(Y_train, y_pred)

            print(accuracy)
            pd.DataFrame(report).transpose()\
                  .to_excel(cfg.DIR_output + "classification_report_" + file_str + '_' + str(rand_num) + ".xlsx")

            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=model_best_para.classes_)
            disp.plot() # plt.show()
            plt.savefig(cfg.DIR_output + 'Confusion_matrix_' + file_str + '_' + str(rand_num) + '.png')
            plt.close()
        else:
            print("The value of model_target_type is not valid: " + str(cfg.model_target_type))

    return model_best_para, L_process_coef, model_path_FName