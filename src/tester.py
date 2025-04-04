import pandas as pd
from sklearn.decomposition import PCA
import xgboost
from tqdm import tqdm
import scipy.stats as stats
import sklearn.metrics as skl_M


import src.utils as utils
import src.data as data


def create_metric(model, X_train, X_test, y_train, y_test):
    train_probPred = model.predict_proba(X_train)[:,1]
    test_probPred = model.predict_proba(X_test)[:,1]

    train_pos_score = train_probPred[y_train.eq(1)]
    train_neg_score = train_probPred[y_train.eq(0)]

    test_pos_score = test_probPred[y_test.eq(1)]
    test_neg_score = test_probPred[y_test.eq(0)]

    df_result = pd.DataFrame()
    df_result = pd.concat(
        [
            df_result,
            pd.DataFrame(
                index=["Train", "Test"],
                data={
                    "Size": [
                        X_train.shape,
                        X_test.shape
                    ],
                    "KS": [
                        stats.ks_2samp(train_pos_score, train_neg_score)[0],
                        stats.ks_2samp(test_pos_score, test_neg_score)[0],
                    ],
                    "AUC": [
                        skl_M.roc_auc_score(y_train, train_probPred),
                        skl_M.roc_auc_score(y_test, test_probPred),
                    ],
                }
            )
        ],
    )
    df_result["GINI"] = 2 * df_result["AUC"] - 1
    return df_result


def validate_model(df, lable_name, map_label=None, **validated_params):
    processed_params = validated_params["processed_params"]
    model_params = validated_params["model_params"]
    df_result = pd.DataFrame()
    for fold in tqdm(range(1, df["fold"].max() + 1)):
        X_train, X_valid, y_train, y_valid = utils.get_fold(
            df=df, fold=fold, lable_name=lable_name,
            map_label=map_label
        )
        X_train, X_valid = data.process_data(X_train, X_valid, y_train, **processed_params)
        # return X_train
        xgb = xgboost.XGBClassifier(**model_params)
        xgb.fit(X_train, y_train)

        df_result = pd.concat(
            [
                df_result,
                create_metric(xgb, X_train, X_valid, y_train, y_valid),
            ],
        )
    return df_result


def test_model(df_train, df_test, lable_name, map_label, **validated_params):
    processed_params = validated_params["processed_params"]
    model_params = validated_params["model_params"]
    df_result = pd.DataFrame()
        
    X_train, X_test, y_train, y_test = utils.get_train_test(df_train, df_test, lable_name, map_label)
    X_train, X_test = data.process_data(X_train, X_test, y_train, **processed_params)

    xgb = xgboost.XGBClassifier(**model_params)
    xgb.fit(X_train, y_train)

    return create_metric(xgb, X_train, X_test, y_train, y_test)